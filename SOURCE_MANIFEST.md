# Source Manifest

This is a reduced public subset prepared from the internal training code on
2026-06-14. Files were renamed and imports were reorganized for publication.

Included:

- the electron-augmented rMD17 dataset loader;
- the current eMol representation and scalar energy/force model;
- the Lightning training module and one example configuration.

Intentionally excluded:

- retrieval and alignment pipelines;
- QM9 and QM9-DFT code;
- MD17, MD22, Chignolin, and Molecule3D loaders;
- alternate and historical model architectures;
- standalone evaluation and data-preparation scripts;
- datasets, logs, checkpoints, and processed artifacts.
