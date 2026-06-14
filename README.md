# eMol

Official implementation of the eMol model for molecular energy and force
prediction using molecular structures and electron-density information.

## Installation

```bash
pip install -r requirements.txt
```

## Training

Configure the dataset paths and training parameters in
`examples/rmd17.yaml`, then run:

```bash
python train_emol.py --conf examples/rmd17.yaml
```
