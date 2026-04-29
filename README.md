# EGNN + TDA

Stepik: 187053348

# Introduction

This work is a final project within the course *"Deep Learning School. Part 1"* in the *Geometric ML* track.

The goal of this project is to build a model based on the EGNN architecture using features derived from Topological Data Analysis (TDA).

The QM9 dataset is used in this study, with the target variable defined as the difference between HOMO and LUMO energy levels (hereafter referred to as `gap`).

---

# Dataset

Several data transformers are implemented in `dataset.py`, providing the following functionality:

- reducing the dataset size using a filter on the maximum number of atoms in a molecule (`MaxAtomsFilter`);
- extracting the target variable and modifying the number of node features (`Add_node_attrs`);
- removing unnecessary dataset attributes (`DropFields`);
- constructing persistence diagrams and converting them into persistence images (PI) via smoothing (`TDA_transform`).

The resulting dataset contains two additional attributes:

- `node_attr`: a collection of all node features;
- `pi`: three-channel arrays corresponding to H0 (connected components) and H1 (cycles/loops).

The atomic charge feature was excluded since it duplicates information already present in the attribute `x`.

Thus, `node_attr` includes the following features:
- H?, C?, N?, O?, F? (categorical indicators),
- atomic charge,
- aromaticity,
- hybridization states (sp, sp2, sp3),
- number of bonded hydrogen atoms.

The symbol “?” denotes categorical features.

---

# Model

A simplified version of the EGNN architecture from the paper  
https://arxiv.org/abs/2102.09844 is implemented in this project.

The model is based on the E(3)-Equivariant Graph Convolution Layer (EGCL), implemented using the `torch_geometric.nn.MessagePassing` class.

The custom `GCL` class takes:
- `hidden_dim` (set to 64 in this work),
- an `equivariant` flag, allowing either invariant or equivariant operation.

For the current task, the invariant version (`equivariant = False`) is sufficient, as no improvement from equivariance was reported in the original paper. Therefore, the invariant model is used.

Within `MessagePassing`, the `message`, `aggregate`, and `forward` functions are redefined according to the paper.

A key advantage of `MessagePassing` is that it automatically restructures inputs from `forward` (`h, x, edge_index, edge_attr`) into the propagation pipeline.

---

Two model variants are implemented:

- `EGNN`
- `EGNN_TDA` (with persistence images)

Both models:
1. start with embedding node and edge attributes,
2. pass data through multiple `GCL` layers (7 in this work),
3. apply several linear layers,
4. perform pooling,
5. and finalize predictions with additional linear layers.

The difference:

- In `EGNN_TDA`, after pooling:
  - a convolutional layer is applied to the persistence image,
  - followed by activation,
  - then a linear layer,
  - and concatenation with pooled graph features.

Thus, persistence images are used as **global descriptors**.

An illustration of H0 and H1 for a sample configuration is shown below:

<img src="figs/pi.png" width="800">

---

# Training and Validation

Training was performed on the full QM9 dataset using the following node features:

H?, C?, N?, O?, F?, aromaticity, sp, sp2, sp3, number of bonded hydrogens.

- Training set: 90%
- Validation set: 10%
- Optimal batch size: 64 (larger batches increase overfitting)

An example training pipeline is provided in:
`notebook/egnn_tda_train.ipynb`

The loss function used is Mean Squared Error (MSE).

The best model is selected based on minimum validation loss.

---

### Target normalization

To improve training stability, the target is normalized as:

y_train = y - y_mean

Thus, to recover the true `gap`, the dataset mean must be added back to the model output.

---

### Loading pretrained models

Pretrained models are available in the `checkpoints` directory:

    ckpt = torch.load("checkpoints/egnn_qm9_gap.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    y_mean = ckpt["mean_y"]

---

# Results

The original paper reports a mean absolute error (MAE) of **48 meV**, which is used here as a reference.

### EGNN

<img src="figs/egnn_acc.png" width="600">

- MAE (train): 91.49 meV  
- MAE (valid): 90.03 meV  

---

### EGNN_TDA

<img src="figs/egnn_tda_acc.png" width="600">

- MAE (train): 71.71 meV  
- MAE (valid): 97.77 meV  

---

### Discussion

Local deviations (outliers) in the training set can be easily reduced by overfitting, but such models have no practical value.

The simpler EGNN model performs slightly better due to lower validation error.

Notably, most validation points lie within the variance of predictions obtained for the training set.

The behavior of the EGNN_TDA model is more complex. Most likely, the convolutional layer introduces additional optimization difficulty under the MSE loss.

It was expected that persistence images would provide corrections related to global molecular structure.

More promising results may be obtained by:

- using larger and more diverse molecular systems,
- avoiding manually fixed Gaussian smoothing,
- incorporating higher-order features such as H2.

Overall, however, the results of both models are comparable.

---

# Environment Setup

    conda env create -f environment.yml
    conda activate egnn_tda