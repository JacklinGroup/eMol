from os.path import join

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from emol_core.config import MissingLabelException, make_splits


class EMolDataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        values = hparams.__dict__ if hasattr(hparams, "__dict__") else hparams
        self.hparams.update(values)
        self._mean = None
        self._std = None
        self._saved_dataloaders = {}

    def prepare_dataset(self):
        dataset_name = self.hparams.get("dataset", "rMD17")
        prepare_fn = getattr(self, f"_prepare_{dataset_name}_dataset", None)
        if prepare_fn is None:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: rMD17, MD17, MD22, QM9"
            )
        prepare_fn()

        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.val_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)
        print(f"train {len(self.idx_train)}, val {len(self.idx_val)}, "
              f"test {len(self.idx_test)}")
        if self.hparams.get("standardize", True):
            self._standardize()

    def _prepare_rMD17_dataset(self):
        from emol_core.dataset.rmd17 import RMD17
        self.dataset = RMD17(
            root=self.hparams["dataset_root"],
            dataset_arg=self.hparams["dataset_arg"],
        )
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.hparams["train_size"], self.hparams["val_size"],
            self.hparams.get("test_size"),
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams.get("splits"),
        )

    def _prepare_MD17_dataset(self):
        from emol_core.dataset.md17 import MD17
        self.dataset = MD17(
            root=self.hparams["dataset_root"],
            dataset_arg=self.hparams["dataset_arg"],
        )
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.hparams["train_size"], self.hparams["val_size"],
            self.hparams.get("test_size"),
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams.get("splits"),
        )

    def _prepare_MD22_dataset(self):
        from emol_core.dataset.md22 import MD22
        self.dataset = MD22(
            root=self.hparams["dataset_root"],
            dataset_arg=self.hparams["dataset_arg"],
        )
        train_val_size = self.dataset.molecule_splits[self.hparams["dataset_arg"]]
        train_size = round(train_val_size * 0.95)
        val_size = train_val_size - train_size
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            train_size, val_size, None,
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams.get("splits"),
        )

    def _prepare_QM9_dataset(self):
        from emol_core.dataset.qm9 import QM9
        self.dataset = QM9(
            root=self.hparams["dataset_root"],
            dataset_arg=self.hparams["dataset_arg"],
        )
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.hparams["train_size"], self.hparams["val_size"],
            self.hparams.get("test_size"),
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams.get("splits"),
        )

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return [self._get_dataloader(self.val_dataset, "val")]

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage, store=True):
        if store and stage in self._saved_dataloaders:
            return self._saved_dataloaders[stage]
        batch_size = (self.hparams["batch_size"] if stage == "train"
                      else self.hparams["inference_batch_size"])
        loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=stage == "train",
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
        )
        if store and not self.hparams["reload"]:
            self._saved_dataloaders[stage] = loader
        return loader

    def _standardize(self):
        def labels(batch):
            if batch.y is None:
                raise MissingLabelException()
            return batch.y.squeeze()
        batches = tqdm(
            self._get_dataloader(self.train_dataset, "val", store=False),
            desc="computing mean and std",
        )
        values = torch.cat([labels(batch) for batch in batches])
        self._mean = values.mean(dim=0)
        self._std = values.std(dim=0)
