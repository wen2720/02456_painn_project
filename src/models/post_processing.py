import torch
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