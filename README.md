# eMol Core Training Code

This repository is a minimal training release for the electron-density-aware
eMol model on electron-augmented rMD17 trajectories.

## Layout

- `train_emol.py`: training entry point.
- `emol_core/`: rMD17 loader and the core eMol model.
- `examples/rmd17.yaml`: example training configuration.

Datasets, electron-density files, processed caches, logs, and checkpoints are
not included.

## Environment

The code was run with Python 3.9 and a CUDA-enabled PyTorch environment.
Install PyTorch, PyTorch Geometric, `torch-scatter`, and `torch-cluster` builds
compatible with the local CUDA toolkit first. Then install the remaining
dependencies.

```bash
pip install -r requirements.txt
```

## Training

```bash
python train_emol.py --conf examples/rmd17.yaml
```

The default configuration expects an rMD17-compatible dataset under
`../data/rmd17_after` and a split file under `../data/splits/`. Adjust
`dataset_root`, `dataset_arg`, `splits`, and `log_dir` in the YAML file for a
new molecule or directory layout.

Each augmented trajectory must contain `nuclear_charges`, `coords`, `energies`,
`forces`, `elec_coords`, and `density`. Run `python train_emol.py --help` for
the available command-line overrides.

## Repository hygiene

The `.gitignore` excludes datasets, processed caches, logs, checkpoints, and
large binary files. This release intentionally excludes retrieval, data
construction, QM9, alternate architectures, and internal evaluation scripts.
