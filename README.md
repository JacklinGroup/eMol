# eMol

**Official PyTorch-based implementation of eMol: Molecular Energy and Force Prediction with Electron Density Augmentation**

eMol is a graph neural network that leverages **electron density information** for accurate molecular energy and force prediction. It constructs a **hybrid atom-electron graph** with four edge types (atom↔atom, atom→electron, electron→atom, electron↔electron) and performs equivariant message passing (lmax ≤ 2) to preserve rotational equivariance for forces.

![Framework](assets/framework.png)

## News!

**\[2025/06\]** Support for MD17, MD22, and QM9 datasets added. RAGED-inspired training configs included.

**\[2025/05\]** eMol open-sourced on GitHub.

## What is eMol?

*Author Note: Molecules are not just static point clouds. The electron density surrounding each atom determines its chemical properties. While conventional GNNs only reason about atomic positions, eMol explicitly **samples tokens from the electron cloud** via farthest-point sampling (FPS) and builds a **heterogeneous graph** where atoms and electron-density tokens exchange messages. This allows the model to learn how electron distribution influences molecular energetics.*

The hybrid graph representation constructed by eMol is as follows:

```
┌──────────────────────────────────────────────────────────────┐
│                    Hybrid Message Passing                    │
│                                                              │
│   Atom Nodes ◉ ←──→ ◉ Atom Nodes   (atom-atom bonds)         │
│        ↕                  ↕                                  │
│   Electron Tokens ⊙ ←──→ ⊙ Electron Tokens  (ee edges)       │
│        ↕                  ↕                                  │
│   Atom → Electron (ae)    Electron → Atom (ea)  (cross edges)│
└──────────────────────────────────────────────────────────────┘
```

When the model is training, it:

1. **Embeds** each atom from its atomic number (z) into a learned hidden vector
2. **Samples** electron cloud points (via FPS) per atom — each atom gets N sample tokens
3. **Projects** electron density features (density, gradients, curvature) into the same hidden space
4. **Builds** the hybrid graph with four edge types within a cutoff radius
5. **Passes messages** through hybrid graph layers — atoms attend to atoms, atoms attend to electron tokens, and vice versa
6. **Gates** the electron information flow with a learnable "electron gate" to control relevance
7. **Outputs** energy (scalar) and forces (gradient of energy w.r.t. positions) via an equivariant output head

## Environments

#### 1. GPU Environment

CUDA ≥ 11.6

Ubuntu 18.04 / 20.04 / 22.04

#### 2. Create Conda Environment

```bash
# create conda env
conda create -n emol python=3.9
conda activate emol

# install pytorch (adjust CUDA version as needed)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install dependencies
pip install -r requirements.txt
```

## Data Processing

#### 1. Data Format

eMol expects `.npz` files containing molecular conformations and (optionally) electron density information. The required fields are:

| Field | Shape | Description |
|---|---|---|
| `nuclear_charges` / `z` | `(N,)` | Atomic numbers of N atoms |
| `coords` / `R` | `(F, N, 3)` | Atomic coordinates for F frames |
| `energies` / `E` | `(F,)` | Total energy per frame |
| `forces` / `F` | `(F, N, 3)` | Atomic forces per frame |
| `elec_coords` | `(F, P, 3)` | Electron grid coordinates (optional) |
| `density` | `(F, P)` | Electron density values (optional) |
| `num_electrons` | `(F,)` | Number of grid points per frame (optional) |

If electron data (`elec_coords`, `density`) is absent, eMol falls back to **atom-only embeddings** and operates like a conventional equivariant GNN.

#### 2. Generating Electron Density Data

The electron density data can be obtained from DFT calculations (e.g., via VASP, CP2K, or Quantum ESPRESSO). We provide a script for processing cube files into the required `.npz` format:

```python
# Example: convert DFT cube data to eMol format
import numpy as np

def process_electron_density(cube_file, coord_file, output_path):
    # Load atomic coordinates and energies
    data = np.load(coord_file)
    
    # Load electron density cube
    # ... (your DFT processing code here)
    
    # Save in eMol format
    np.savez(output_path,
             nuclear_charges=atomic_numbers,
             coords=positions,
             energies=energies,
             forces=forces,
             elec_coords=electron_grid_points,
             density=electron_density_values,
             num_electrons=np.array([len(electron_grid_points)]))
```

#### 3. Dataset Structure

Organize your data as follows:

```text
data/
├── rmd17/                          # rMD17 dataset
│   ├── rmd17_ethanol.npz
│   ├── rmd17_aspirin.npz
│   └── ...
├── md17/                           # MD17 dataset (original)
│   ├── md17_ethanol.npz
│   └── ...
├── md22/                           # MD22 dataset
│   ├── md22_AT-AT.npz
│   └── ...
└── splits/                         # Train/val/test splits
    └── splits.npz
```

#### 4. Supported Datasets

| Dataset | Molecules / Targets | Reference |
|---|---|---|
| **rMD17** | aspirin, azobenzene, benzene, ethanol, malonaldehyde, naphthalene, paracetamol, salicylic, toluene, uracil | [Materials Cloud](https://archive.materialscloud.org/record/2020.82) |
| **MD17** | aspirin, ethanol, malonaldehyde, naphthalene, salicylic_acid, toluene, uracil | [QM-GDML](http://www.quantum-machine.org/gdml/#datasets) |
| **MD22** | Ac_Ala3_NHMe, DHA, stachyose, AT_AT, AT_AT_CG_CG, buckyball_catcher, double_walled_nanotube | [QM-GDML](http://www.quantum-machine.org/gdml/#datasets) |
| **QM9** | mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv | [PyG QM9](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html) |
| **EDBench** | MD17, MD22, QM9 with electron density | [GitHub](https://github.com/HongxinXiang/EDBench) |

## 🔥Pre-training

eMol is trained from scratch on molecular dynamics trajectories with energy+force supervision. The model jointly optimizes energy (MSE) and force (MSE/l1) losses.

#### 1. Preparing Dataset for Training

Download the rMD17 dataset from [Materials Cloud](https://archive.materialscloud.org/record/2020.82) and place the `.npz` files in your data directory.

```bash
mkdir -p data/rmd17
# Download and extract rMD17 npz files into data/rmd17/
```

Alternatively, provide your own dataset in the format described above.

#### 2. Train eMol

Usage:

```
usage: train_emol.py [-h] [--load-model LOAD_MODEL] [--conf CONF]
                     [--dataset {rMD17,MD17,MD22,QM9}]
                     [--dataset-root DATASET_ROOT]
                     [--dataset-arg DATASET_ARG]
                     [--model {EMolRepresentation,RAGEDSampledBlock}]
                     [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE]
                     [--lr LR] [--num-layers NUM_LAYERS]
                     [--embedding-dimension EMBEDDING_DIMENSION]
                     [--num-sample-points NUM_SAMPLE_POINTS]
                     [--electron-gate-mode {fixed,trainable,schedule,plateau,lr_scale}]
                     ...
```

**Quick start (rMD17):**

```bash
python train_emol.py -c examples/rmd17.yaml
```

**RAGED-style training (electron-augmented):**

```bash
python train_emol.py -c examples/RAGEDSampled-rMD17.yml
```

**On different datasets:**

```bash
# MD22
python train_emol.py -c examples/RAGED-MD22.yml

# MD17 (original)
python train_emol.py -c examples/RAGEDSampled-MD17.yml
```

#### 3. Training with CLI Arguments

All configuration can also be passed directly via command-line:

```bash
python train_emol.py \
  --dataset rMD17 \
  --dataset-root /path/to/data \
  --dataset-arg ethanol \
  --model RAGEDSampledBlock \
  --batch-size 4 \
  --num-layers 9 \
  --embedding-dimension 256 \
  --num-sample-points 3 \
  --electron-gate-mode trainable \
  --electron-gate 0.5 \
  --aeea-five-body-scale 0.05 \
  --log-dir logs/ethanol \
  --seed 1
```

#### 4. Training Configuration Reference

**Training settings:**

| Parameter | Default | Description |
|---|---|---|
| `num_epochs` | 3000 | Number of training epochs |
| `lr` | 2e-4 | Initial learning rate |
| `lr_warmup_steps` | 1000 | Linear LR warmup steps |
| `lr_patience` | 30 | ReduceLROnPlateau patience |
| `lr_factor` | 0.8 | LR reduction factor on plateau |
| `lr_min` | 1e-7 | Minimum learning rate |
| `weight_decay` | 0.0 | AdamW weight decay |
| `early_stopping_patience` | 500 | Early stopping patience (epochs) |
| `loss_type` | MSE | Loss function (MSE or MAE) |
| `energy_weight` | 0.05 | Weight for energy term in total loss |
| `force_weight` | 0.95 | Weight for force term in total loss |
| `gradient_clip_val` | 1.0 | Gradient clipping threshold |
| `gradient_clip_algorithm` | norm | Clipping method (norm or value) |

**Model architecture:**

| Parameter | Default | Description |
|---|---|---|
| `model` | EMolRepresentation | Model variant |
| `embedding_dimension` | 256 | Hidden feature dimension |
| `num_layers` | 9 | Number of message-passing layers |
| `num_heads` | 8 | Multi-head attention heads |
| `num_rbf` | 32 | Number of radial basis functions |
| `rbf_type` | expnorm | RBF type (expnorm or gauss) |
| `cutoff` | 5.0 | Interaction cutoff radius (Å) |
| `max_num_neighbors` | 32 | Maximum neighbors per node |
| `lmax` | 2 | Maximum spherical harmonics degree (equivariant) |
| `activation` | silu | Activation function |
| `reduce_op` | add | Atom-wise aggregation (add or mean) |

**Electron density module:**

| Parameter | Default | Description |
|---|---|---|
| `electron_radius` | 2.0 | Radius (Å) for electron cloud sampling |
| `learnable_radius` | True | Whether the sampling radius is learnable |
| `num_sample_points` | 3 | Number of electron tokens sampled per atom |
| `atom_token_extra_neighbors` | 1 | Extra atom neighbors connected per electron token |
| `electron_gate` | 0.25 | Initial value for electron information gate |
| `electron_gate_mode` | fixed | Gate mode: `fixed`, `trainable`, `schedule`, `plateau`, `lr_scale` |
| `aeea_five_body_scale` | 0.0 | Cross-branch (atom↔electron) residual scale |
| `aeea_five_body_use_gate` | True | Apply gate to cross-branch residuals |
| `ee_cutoff` | null | Electron-electron cutoff (defaults to `cutoff`) |
| `ee_scalar_scale` | 0.1 | Electron-electron scalar message scale |
| `ee_vector_scale` | 0.05 | Electron-electron vector message scale |

**Data:**

| Parameter | Default | Description |
|---|---|---|
| `dataset` | rMD17 | Dataset name |
| `dataset_root` | — | Path to data directory |
| `dataset_arg` | — | Molecule/target name |
| `derivative` | True | Whether to compute force predictions |
| `splits` | null | Path to precomputed `.npz` split file |
| `train_size` | 950 | Training set size |
| `val_size` | 50 | Validation set size |
| `batch_size` | 4 | Training batch size |
| `inference_batch_size` | 16 | Eval/test batch size |
| `num_workers` | 6 | DataLoader workers |

## 🔥Fine-tuning / Inference

#### 1. Using a Pretrained Model

eMol can continue training from a checkpoint or run inference:

```bash
# Continue training from checkpoint
python train_emol.py \
  -c examples/rmd17.yaml \
  --load-model /path/to/checkpoint.ckpt

# Inference only (requires a test set)
python train_emol.py \
  -c examples/rmd17.yaml \
  --task inference \
  --load-model /path/to/checkpoint.ckpt
```

#### 2. Output Structure

Training logs and checkpoints are saved to `--log-dir`:

```
logs/ethanol/
├── input.yaml                    # Saved configuration
├── last.ckpt                     # Last checkpoint (resume training)
├── epoch=XXX-val_loss=YYYY.ckpt  # Best checkpoint
├── tensorboard/                  # TensorBoard event files
├── metrics.csv                   # Per-epoch metrics (CSV)
└── ratio_metrics.csv             # AE/EA ratio log (for hybrid models)
```

#### 4. Performance on MD17

MD17 energy (kcal/mol) and force (kcal/mol/Å) MAE results compared with state-of-the-art methods (lower is better).

| Molecule | Target | SchNet | DimeNet | PaiNN | ET | ViSNet | QuinNet | PaiNN-TAIP | LiTEN | MGNN | **eMol** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Aspirin | Energy | 0.37 | 0.204 | 0.167 | 0.123 | 0.116 | 0.119 | 0.144 | 0.114 | 0.126 | **0.113** |
| Aspirin | Forces | 1.35 | 0.499 | 0.338 | 0.253 | 0.155 | 0.145 | 0.33 | 0.152 | 0.181 | **0.149** |
| Ethanol | Energy | 0.08 | 0.064 | 0.064 | 0.052 | 0.051 | 0.05 | 0.061 | 0.049 | 0.057 | **0.046** |
| Ethanol | Forces | 0.39 | 0.23 | 0.224 | 0.109 | 0.06 | 0.06 | 0.18 | 0.059 | 0.071 | **0.057** |
| Malondialdehyde | Energy | 0.13 | 0.104 | 0.091 | 0.077 | 0.075 | 0.078 | 0.091 | 0.074 | 0.083 | **0.073** |
| Malondialdehyde | Forces | 0.66 | 0.383 | 0.319 | 0.169 | 0.1 | 0.097 | 0.297 | 0.098 | 0.118 | **0.095** |
| Naphthalene | Energy | 0.16 | 0.122 | 0.116 | 0.085 | 0.085 | 0.101 | 0.093 | 0.083 | 0.092 | **0.082** |
| Naphthalene | Forces | 0.58 | 0.215 | 0.077 | 0.061 | 0.039 | 0.039 | 0.072 | 0.038 | 0.051 | **0.037** |
| Salicylic acid | Energy | 0.2 | 0.134 | 0.116 | 0.093 | 0.092 | 0.101 | 0.105 | 0.091 | 0.101 | **0.09** |
| Salicylic acid | Forces | 0.85 | 0.374 | 0.195 | 0.129 | 0.08 | 0.08 | 0.193 | 0.079 | 0.096 | **0.08** |
| Toluene | Energy | 0.12 | 0.102 | 0.095 | 0.074 | 0.074 | 0.08 | 0.109 | 0.072 | 0.081 | **0.071** |
| Toluene | Forces | 0.57 | 0.216 | 0.094 | 0.067 | 0.039 | 0.039 | 0.09 | 0.041 | 0.055 | **0.042** |
| Uracil | Energy | 0.14 | 0.115 | 0.106 | 0.095 | 0.095 | 0.096 | 0.09 | 0.091 | 0.099 | **0.092** |
| Uracil | Forces | 0.56 | 0.301 | 0.139 | 0.095 | 0.062 | 0.062 | 0.13 | 0.057 | 0.073 | **0.055** |

#### 5. Performance on QM9

QM9 MAE results compared with state-of-the-art methods (lower is better). eMol achieves competitive performance on energy, force, and electronic property prediction.

| Target | Unit | SchNet | EGNN | DimeNet++ | PaiNN | SphereNet | PaxNet | ET | ComENet | ViSNet | QuinNet | LiTEN | MGNN | **eMol** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| μ | D | 33 | 29 | 29.7 | 12 | 24.5 | 10.8 | 11 | 24.5 | 9.5 | 771 | 9.8 | 10 | **10.8** |
| α | a₀³ | 235 | 71 | 43.5 | 45 | 44.9 | 44.7 | 59 | 45.2 | 41.1 | 47 | 40.8 | 41 | **40.5** |
| ε<sub>HOMO</sub> | meV | 41 | 29 | 24.6 | 27.6 | 22.8 | 22.8 | 20.3 | 23.1 | 17.3 | 20.4 | 16.9 | 23.2 | **16.2** |
| ε<sub>LUMO</sub> | meV | 34 | 25 | 19.5 | 20.4 | 18.9 | 19.2 | 17.5 | 19.8 | 14.8 | 17.6 | 14.6 | 17 | **14.3** |
| Δε | meV | 63 | 48 | 32.6 | 45.7 | 31.1 | 31 | 36.1 | 32.4 | 31.7 | 28.2 | 28.5 | 30 | **27.8** |
| ⟨R²⟩ | a₀² | 73 | 106 | 331 | 66 | 268 | 93 | 33 | 259 | 29.8 | 194 | 30.1 | 40 | **30.4** |
| ZPVE | meV | 1.7 | 1.55 | 1.21 | 1.28 | 1.12 | 1.17 | 1.84 | 1.2 | 1.56 | 1.26 | 1.08 | 1.17 | **1.05** |
| U₀ | meV | 14 | 11 | 6.32 | 5.85 | 6.26 | 5.9 | 6.15 | 6.59 | 4.23 | 7.6 | 4.18 | 4.1 | **4.29** |
| U | meV | 19 | 12 | 6.28 | 5.83 | 6.36 | 5.92 | 6.38 | 6.82 | 4.25 | 8.4 | 4.22 | 4.2 | **4.33** |
| H | meV | 14 | 12 | 6.53 | 5.98 | 6.33 | 6.04 | 6.16 | 6.86 | 4.52 | 7.8 | 4.15 | 4.1 | **4.08** |
| G | meV | 14 | 12 | 7.56 | 7.35 | 7.78 | 7.14 | 7.62 | 7.98 | 5.86 | 8.5 | 5.76 | 5.7 | **5.82** |
| Cᵥ | mcal/(mol·K) | 33 | 31 | 23 | 24 | 22 | 23.1 | 26 | 24 | 23 | – | 22.8 | 23 | **23** |

## Reproducing Guidance

Clone the repository, set up the environment, download the data, and run the provided configuration files:

```bash
# 1. Clone
git clone https://github.com/YangSun1/eMol.git
cd eMol

# 2. Environment
conda create -n emol python=3.9
conda activate emol
pip install -r requirements.txt

# 3. Data
# Download rMD17 from https://archive.materialscloud.org/record/2020.82
# Place .npz files under data/rmd17/

# 4. Train
python train_emol.py -c examples/rmd17.yaml

# 5. Monitor
tensorboard --logdir logs/
```

## Model Variants

| Model Name | Description | Electron Sampling | Graph Type |
|---|---|---|---|
| `EMolRepresentation` | Default eMol model | Farthest-point sampling (FPS) | Hybrid (AA + AE + EA + EE) |
| `RAGEDSampledBlock` | Alias for EMolRepresentation | FPS with kernel weighting | Hybrid (AA + AE + EA + EE) |

## Acknowledge

We would like to thank the following useful tools and data:

\[1\] Wang Y, Wang T, Li S, et al. "Enhancing geometric representations for molecules with equivariant vector-scalar interactive message passing." Nature Communications, 15, 313 (2024). [DOI: 10.1038/s41467-023-43720-2](https://doi.org/10.1038/s41467-023-43720-2).

\[2\] Chmiela S, Tkatchenko A, Sauceda H E, Poltavsky I, Schütt K T, Müller K R. "Machine learning of accurate energy-conserving molecular force fields." Science Advances, 3(5), e1603015 (2017). [DOI: 10.1126/sciadv.1603015](https://doi.org/10.1126/sciadv.1603015).

\[3\] Christensen A S, von Lilienfeld O A. "On the role of gradients for machine learning of molecular energies and forces." Machine Learning: Science and Technology, 1(4), 045018, 2020. [DOI: 10.1088/2632-2153/abba6f](https://doi.org/10.1088/2632-2153/abba6f).

\[4\] Chmiela S, Vassilev-Galindo V, Unke O T, Kabylda A, Sauceda H E, Tkatchenko A, Müller K-R. "Accurate global machine learning force fields for molecules with hundreds of atoms." Science Advances, 9(32), eadf0873, 2023. [DOI: 10.1126/sciadv.adf0873](https://doi.org/10.1126/sciadv.adf0873).

\[5\] Schütt K T, Unke O, Gastegger M. "Equivariant message passing for the prediction of tensorial properties and molecular spectra." Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139:9377–9388, 2021. [https://proceedings.mlr.press/v139/schutt21a.html](https://proceedings.mlr.press/v139/schutt21a.html).

\[6\] Xiang H, Li K, Liu M, et al. "EDBench: Large-Scale Electron Density Data for Molecular Modeling." NeurIPS Datasets and Benchmarks Track, 2025. [OpenReview](https://openreview.net/forum?id=pAd7qVrYPG).

## License

This project is licensed under the MIT License.
