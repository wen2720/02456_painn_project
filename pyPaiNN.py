import torch
import argparse
from tqdm import trange
import torch.nn.functional as F
from pytorch_lightning import seed_everything


import numpy as np
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from typing import Optional, List, Union, Tuple
from torch_geometric.transforms import BaseTransform


class GetTarget(BaseTransform):
    def __init__(self, target: Optional[int] = None) -> None:
        self.target = [target]


    def forward(self, data: Data) -> Data:
        if self.target is not None:
            data.y = data.y[:, self.target]
        return data


class QM9DataModule(pl.LightningDataModule):

    target_types = ['atomwise' for _ in range(19)]
    target_types[0] = 'dipole_moment'
    target_types[5] = 'electronic_spatial_extent'

    # Specify unit conversions (eV to meV).
    unit_conversion = {
        i: (lambda t: 1000*t) if i not in [0, 1, 5, 11, 16, 17, 18]
        else (lambda t: t)
        for i in range(19)
    }

    def __init__(
        self,
        target: int = 7,
        data_dir: str = 'data/',
        batch_size_train: int = 100,
        batch_size_inference: int = 1000,
        num_workers: int = 0,
        splits: Union[List[int], List[float]] = [110000, 10000, 10831],
        seed: int = 0,
        subset_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.target = target
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.subset_size = subset_size

        self.data_train = None
        self.data_val = None
        self.data_test = None


    def prepare_data(self) -> None:
        # Download data
        QM9(root=self.data_dir)


    def setup(self, stage: Optional[str] = None) -> None:
        dataset = QM9(root=self.data_dir, transform=GetTarget(self.target))

        # Shuffle dataset
        rng = np.random.default_rng(seed=self.seed)
        dataset = dataset[rng.permutation(len(dataset))]

        # Subset dataset
        if self.subset_size is not None:
            dataset = dataset[:self.subset_size]
        
        # Split dataset
        if all([type(split) == int for split in self.splits]):
            split_sizes = self.splits
        elif all([type(split) == float for split in self.splits]):
            split_sizes = [int(len(dataset) * prop) for prop in self.splits]

        split_idx = np.cumsum(split_sizes)
        self.data_train = dataset[:split_idx[0]]
        self.data_val = dataset[split_idx[0]:split_idx[1]]
        self.data_test = dataset[split_idx[1]:]


    def get_target_stats(
        self,
        remove_atom_refs: bool = True,
        divide_by_atoms: bool = True
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        atom_refs = self.data_train.atomref(self.target)

        ys = list()
        for batch in self.train_dataloader(shuffle=False):
            y = batch.y.clone()
            if remove_atom_refs and atom_refs is not None:
                y.index_add_(
                    dim=0, index=batch.batch, source=-atom_refs[batch.z]
                )
            if divide_by_atoms:
                _, num_atoms  = torch.unique(batch.batch, return_counts=True)
                y = y / num_atoms.unsqueeze(-1)
            ys.append(y)

        y = torch.cat(ys, dim=0)
        return y.mean(), y.std(), atom_refs


    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )


    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


import torch.nn as nn

class AtomwisePostProcessing(nn.Module):
    """
    Post-processing for (QM9) properties that are predicted as sums of atomic
    contributions.
    """
    def __init__(
        self,
        num_outputs: int,
        mean: torch.FloatTensor,
        std: torch.FloatTensor,
        atom_refs: torch.FloatTensor,
    ) -> None:
        """
        Args:
            num_outputs: Integer with the number of model outputs. In most
                cases 1.
            mean: torch.FloatTensor with mean value to shift atomwise
                contributions by.
            std: torch.FloatTensor with standard deviation to scale atomwise
                contributions by.
            atom_refs: torch.FloatTensor of size [num_atom_types, 1] with
                atomic reference values.
        """
        super().__init__()
        self.num_outputs = num_outputs
        self.register_buffer('scale', std)
        self.register_buffer('shift', mean)
        self.atom_refs = nn.Embedding.from_pretrained(atom_refs, freeze=True)


    def forward(
        self,
        atomic_contributions: torch.FloatTensor,
        atoms: torch.LongTensor,
        graph_indexes: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Atomwise post-processing operations and atomic sum.

        Args:
            atomic_contributions: torch.FloatTensor of size [num_nodes,
                num_outputs] with each node's contribution to the overall graph
                prediction, i.e., each atom's contribution to the overall
                molecular property prediction.
            atoms: torch.LongTensor of size [num_nodes] with atom type of each
                node in the graph.
            graph_indexes: torch.LongTensor of size [num_nodes] with the graph 
                index each node belongs to.

        Returns:
            A torch.FLoatTensor of size [num_graphs, num_outputs] with
            predictions for each graph (molecule).
        """
        num_graphs = torch.unique(graph_indexes).shape[0]

        atomic_contributions = atomic_contributions*self.scale + self.shift
        atomic_contributions = atomic_contributions + self.atom_refs(atoms)

        # Sum contributions for each graph
        output_per_graph = torch.zeros(
            (num_graphs, self.num_outputs),
            device=atomic_contributions.device,
        )
        output_per_graph.index_add_(
            dim=0,
            index=graph_indexes,
            source=atomic_contributions,
        )

        return output_per_graph
    
import torch.nn as nn
from torch.nn import Linear, SiLU
from torch_scatter import scatter_sum

class Message(nn.Module):
    def __init__(self, Ls=None, Lrbf=None, nRbf=20, nF=128, rCut=5.0):
        super(Message, self).__init__()
        self.Ls = Ls if Ls is not None else nn.Sequential(
            Linear(nF, nF),
            SiLU(),
            Linear(nF, 3*nF),
        )
        self.Lrbf = Lrbf if Lrbf is not None else Linear(nRbf, 3*nF)
        self.nRbf = nRbf
        self.rCut = rCut

    def fCut(self, rij_norm, rCut):
        f_cut = 0.5 * (torch.cos(torch.pi * rij_norm / rCut) + 1)
        f_cut[rij_norm > rCut] = 0 
        return f_cut

    def fRBF(self, rij_norm, rCut, nRbf):
        Trbf = torch.arange(1, nRbf + 1, device=rij_norm.device).float()
        rij_norm = rij_norm.unsqueeze(-1)  
        RBF = torch.sin(Trbf * torch.pi * rij_norm / rCut) / (rij_norm + 1e-8)
        return RBF

    def forward(self, vj, sj, rij_vec, eij):
        rij_norm = torch.norm(rij_vec, dim=-1)
        rij_hat =  rij_vec / (rij_norm.unsqueeze(-1) + 1e-8)

        RBF = self.fRBF(rij_norm, self.rCut, self.nRbf)
        T_RBF = self.Lrbf(RBF)
        Ws = T_RBF * self.fCut(rij_norm,self.rCut).unsqueeze(-1) 

        phi = self.Ls(sj)
        phiW = phi * Ws

        SPLIT1 = phiW[:,0:128]
        SPLIT2 = phiW[:,128:256]
        SPLIT3 = phiW[:,256:]

        phiWvv = vj * SPLIT1.unsqueeze(-1).repeat(1, 1, 3)
        phiWvs = SPLIT3.unsqueeze(-1) * rij_hat.unsqueeze(1)
        
        d_vim = scatter_sum((phiWvv + phiWvs), eij[1], dim=0)
        d_sim = scatter_sum(SPLIT2, eij[1], dim=0)
        return d_vim, d_sim

class Update(nn.Module):
    def __init__(self, Luu=None, Luv=None, Ls=None):
        super(Update, self).__init__()
        self.Luu = Luu if Luu is not None else Linear(3, 3, False)
        self.Luv = Luv if Luv is not None else Linear(3, 3, False)
        
        self.Ls = Ls if Ls is not None else nn.Sequential(
            Linear(in_features=256, out_features=128),
            SiLU(),
            Linear(in_features=128, out_features=384),
        )

    def forward(self, vi, si):
        Uvi = self.Luu(vi) 
        Vvi = self.Luv(vi)

        V_norm = torch.norm(Vvi,dim=-1)
        STACK = torch.hstack([V_norm, si])

        SP = torch.sum(Uvi * Vvi, dim=-1) 

        SPLIT = self.Ls(STACK)
        SPLIT1 = SPLIT[:, 0:128]
        SPLIT2 = SPLIT[:, 128:256]
        SPLIT3 = SPLIT[:, 256:]

        d_viu = Uvi * SPLIT1.unsqueeze(-1).repeat(1, 1, 3)
        d_siu = SP * SPLIT2 + SPLIT3

        return d_viu, d_siu

from torch_geometric.nn import radius_graph

class PaiNN(nn.Module):
    """
    Polarizable Atom Interaction Neural Network with PyTorch.
    """
    def __init__(
        self, Lm, Lu,
        num_message_passing_layers: int = 3,
        num_features: int = 128,
        num_outputs: int = 1,
        num_rbf_features: int = 20,
        num_unique_atoms: int = 100,
        cutoff_dist: float = 5.0,
    ) -> None:
        """
        Args:
            num_message_passing_layers: Number of message passing layers in
                the PaiNN model.
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_outputs: Number of model outputs. In most cases 1.
            num_rbf_features: Number of radial basis functions to represent
                distances.
            num_unique_atoms: Number of unique atoms in the data that we want
                to learn embeddings for.
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        #raise NotImplementedError
        self.num_message_passing_layers = num_message_passing_layers
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_rbf_features = num_rbf_features
        self.num_unique_atoms = num_unique_atoms
        self.cutoff_dist = cutoff_dist

        self.zi = nn.Embedding(num_unique_atoms, num_features)

        self.Lm = Lm
        self.Lu = Lu

        self.Lr = nn.Sequential(
            Linear(in_features=128, out_features=64),
            SiLU(),
            Linear(in_features=64, out_features=1),
        )

    def forward(
        self,
        atoms: torch.LongTensor,
        atom_positions: torch.FloatTensor,
        graph_indexes: torch.LongTensor,
    ) -> torch.FloatTensor:
        si = self.zi(atoms)
        eij = radius_graph(atom_positions, r=self.cutoff_dist, batch=graph_indexes)
        sj = si[eij[0]]
        vi = torch.zeros_like(si).unsqueeze(-1).repeat(1, 1, 3)
        vj = vi[eij[0]]
        rij_vec = atom_positions[eij[0]] - atom_positions[eij[1]]
        for _ in range(self.num_message_passing_layers):
            d_vim, d_sim = self.Lm(vj, sj, rij_vec, eij)
            vi = vi + d_vim
            si = si + d_sim

            d_viu, d_siu = self.Lu(vi, si)

            vi = vi + d_viu
            si = si + d_siu
        
        Sigma = self.Lr(si)

        return Sigma
    

def cli(args: list = []):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0)

    # Data
    parser.add_argument('--target', default=7, type=int) # 7 => Internal energy at 0K
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--batch_size_train', default=100, type=int)
    parser.add_argument('--batch_size_inference', default=1000, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--splits', nargs=3, default=[110000, 10000, 10831], type=int) # [num_train, num_val, num_test]
    parser.add_argument('--subset_size', default=None, type=int)

    # Model
    parser.add_argument('--num_message_passing_layers', default=3, type=int)
    parser.add_argument('--num_features', default=128, type=int)
    parser.add_argument('--num_outputs', default=1, type=int)
    parser.add_argument('--num_rbf_features', default=20, type=int)
    parser.add_argument('--num_unique_atoms', default=100, type=int)
    parser.add_argument('--cutoff_dist', default=5.0, type=float)

    # Training    
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-8, type=float)
    parser.add_argument('--num_epochs', default=1000, type=int)

    args = parser.parse_args(args=args)
    return args




args = [] # Specify non-default arguments in this list
args = cli(args)
seed_everything(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
device_name = torch.cuda.get_device_name(torch.cuda.current_device())
print(device_name)
dm = QM9DataModule(
    target=args.target,
    data_dir=args.data_dir,
    batch_size_train=args.batch_size_train,
    batch_size_inference=args.batch_size_inference,
    num_workers=args.num_workers,
    splits=args.splits,
    seed=args.seed,
    subset_size=args.subset_size,
)
dm.prepare_data()
dm.setup()
y_mean, y_std, atom_refs = dm.get_target_stats(
    remove_atom_refs=True, divide_by_atoms=True
)

painn = PaiNN(
    Lm=Message(),
    Lu=Update(),
    num_message_passing_layers=args.num_message_passing_layers,
    num_features=args.num_features,
    num_outputs=args.num_outputs, 
    num_rbf_features=args.num_rbf_features,
    num_unique_atoms=args.num_unique_atoms,
    cutoff_dist=args.cutoff_dist,
)
post_processing = AtomwisePostProcessing(
    args.num_outputs, y_mean, y_std, atom_refs
)

painn.to(device)
post_processing.to(device)

import torch.optim as optim
optimizer = optim.AdamW(painn.parameters(), lr=args.lr,weight_decay=args.weight_decay)
#optimizer = optim.SGD(painn.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


train_losses, val_losses, val_maes = [], [], []
best_val_loss = float('inf')
patience = 30  # Number of epochs to wait before stopping



smoothed_val_loss = None
smoothing_factor = 0.9
wait = 0

plateau_patience = 5
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=plateau_patience, threshold=1e-4
)

from torch.optim.swa_utils import AveragedModel, SWALR
swa_model = AveragedModel(painn)
swa_scheduler = SWALR(optimizer, swa_lr=args.lr)

print(args)
print(painn)

import csv
iCsv = "training_log.csv"

with open(iCsv, mode="w", newline="") as file:
    writer = csv.writer(file)  
    writer.writerow(["Epoch", "Training loss", "Validation loss", "smoothed_val_loss", "current_lr"])

painn.train()
for epoch in range(args.num_epochs):        

    loss_epoch = 0.
    for batch in dm.train_dataloader():
        batch = batch.to(device)

        atomic_contributions = painn(
            atoms=batch.z,
            atom_positions=batch.pos,
            graph_indexes=batch.batch
        )
        preds = post_processing(
            atoms=batch.z,
            graph_indexes=batch.batch,
            atomic_contributions=atomic_contributions,
        )
        loss_step = F.mse_loss(preds, batch.y, reduction='sum')
        loss = loss_step / len(batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_epoch += loss_step.detach().item()

    loss_epoch /= len(dm.data_train)
    train_losses.append(loss_epoch)

    painn.eval()
    val_loss_epoch = 0.0
    with torch.no_grad():
        for batch in dm.val_dataloader():
            batch = batch.to(device)

            atomic_contributions = painn(
                atoms=batch.z,
                atom_positions=batch.pos,
                graph_indexes=batch.batch,
            )
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )
            loss_step = F.mse_loss(preds, batch.y, reduction='sum')

            val_loss_epoch += loss_step.item()

    val_loss_epoch /= len(dm.data_val)
    val_losses.append(val_loss_epoch)
    

    if smoothed_val_loss is None:
        smoothed_val_loss = val_loss_epoch
    else:
        smoothed_val_loss = smoothing_factor * val_loss_epoch + (1 - smoothing_factor) * smoothed_val_loss

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch: {epoch + 1}\tTL: {loss_epoch:.3e}\tVL: {val_loss_epoch:.3e}\tsmoothed_val_loss: {smoothed_val_loss} current_lr{current_lr}")
    with open(iCsv, mode="a", newline="") as file:
        writer = csv.writer(file)  
        writer.writerow([epoch + 1, loss_epoch, val_loss_epoch, smoothed_val_loss, current_lr])
    
    # Early Stopping
    if smoothed_val_loss < best_val_loss:
        best_val_loss = smoothed_val_loss
        wait = 0  
        torch.save(painn.state_dict(), "better_painn.pth")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    if epoch >= 140:
        print(f"SWA: triggered after {epoch + 1} epochs.")
        swa_model.update_parameters(painn)
        swa_scheduler.step()
    else:
        scheduler.step(smoothed_val_loss)
    

painn.load_state_dict(torch.load("better_painn.pth", weights_only=True))
mae = 0
painn.eval()
with torch.no_grad():
    for batch in dm.test_dataloader():
        batch = batch.to(device)

        atomic_contributions = painn(
            atoms=batch.z,
            atom_positions=batch.pos,
            graph_indexes=batch.batch,
        )
        preds = post_processing(
            atoms=batch.z,
            graph_indexes=batch.batch,
            atomic_contributions=atomic_contributions,
        )
        mae += F.l1_loss(preds, batch.y, reduction='sum')

mae /= len(dm.data_test)
unit_conversion = dm.unit_conversion[args.target]
print(f'Test MAE: {unit_conversion(mae):.3f}')


torch.optim.swa_utils.update_bn(dm.train_dataloader(), swa_model)
swa_mae = 0
swa_model.eval()
with torch.no_grad():
    for batch in dm.test_dataloader():
        batch = batch.to(device)

        atomic_contributions = painn(
            atoms=batch.z,
            atom_positions=batch.pos,
            graph_indexes=batch.batch,
        )
        preds = post_processing(
            atoms=batch.z,
            graph_indexes=batch.batch,
            atomic_contributions=atomic_contributions,
        )
        swa_mae += F.l1_loss(preds, batch.y, reduction='sum')

swa_mae /= len(dm.data_test)
unit_conversion = dm.unit_conversion[args.target]
print(f'swa Test MAE: {unit_conversion(mae):.3f}')


# Plot Training and Validation Metrics
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Metrics")
#plt.show()
plt.savefig("train_val_loss.png")
plt.close()