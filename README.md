# 02456 Molecular Property Prediction using Graph Neural Networks
This repository serves as a help to get you started with the project "Molecular Property Prediction using Graph Neural Networks" in 02456 Deep Learning. In particular, it provides code for loading and using the QM9 dataset and for post-processing [PaiNN](https://arxiv.org/pdf/2102.03150)'s atomwise predictions as well as a minimal example of how to train the PaiNN model using these modules.

The repository should only be seen as a help and not a limitation in any way. You are free to modify and extend the code in any way you see fit or not use it all.

## Setup
To setup the code environment execute the following commands:
1. `git clone git@github.com:jonasvj/02456_painn_project.git`
2. `cd 02456_painn_project/`
3. `conda env create -f environment.yml`
4. `conda activate painn`
5. `pip install -e .`


## Provided modules
1. `src.data.QM9DataModule`: A PyTorch Lightning datamodule that you can use for loading and processing the QM9 dataset.
2. `src.models.AtomwisePostProcessing`: A module for performing post-processing of PaiNN's atomwise predictions. These steps can be slightly tricky/confusing but are necessary to achieve good performance (compared to normal standardization of target variables.)


## Usage
1. Implement the PaiNN model (`src.models.PaiNN`)
2. Run `python3 minimal_example.py`
3. To achieve similar performance to Schütt et al. (2021), you will most likely need to implement the training "tricks" used in the original paper, e.g., early-stopping, learning rate decay, etc.