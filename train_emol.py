import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from emol_core.config import LoadFromCheckpoint, LoadFromFile, number, save_argparse
from emol_core.data_module import EMolDataModule
from emol_core.model.geometry import act_class_mapping, rbf_class_mapping
from emol_core.training_module import EMolTask


def get_args():
    parser = argparse.ArgumentParser(description="Train eMol on electron-augmented rMD17.")
    parser.add_argument("--load-model", action=LoadFromCheckpoint)
    parser.add_argument("--conf", "-c", type=open, action=LoadFromFile)

    parser.add_argument("--num-epochs", type=int, default=3000)
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-patience", type=int, default=30)
    parser.add_argument("--lr-min", type=float, default=1e-7)
    parser.add_argument("--lr-factor", type=float, default=0.8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=500)
    parser.add_argument("--loss-type", choices=["MSE", "MAE"], default="MSE")
    parser.add_argument("--loss-scale-y", type=float, default=0.1)
    parser.add_argument("--loss-scale-dy", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.25)
    parser.add_argument("--force-weight", type=float, default=0.75)

    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset-arg", default=None)
    parser.add_argument("--derivative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--train-size", type=number, default=950)
    parser.add_argument("--val-size", type=number, default=50)
    parser.add_argument("--test-size", type=number, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--reload", type=int, default=0)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--model", default="EMolRepresentation", choices=["EMolRepresentation"])
    parser.add_argument("--output-model", default="Scalar", choices=["Scalar"])
    parser.add_argument("--prior-model", default=None)
    parser.add_argument("--prior-args", default=None)
    parser.add_argument("--embedding-dimension", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--num-rbf", type=int, default=32)
    parser.add_argument("--activation", choices=list(act_class_mapping), default="silu")
    parser.add_argument("--rbf-type", choices=list(rbf_class_mapping), default="expnorm")
    parser.add_argument("--trainable-rbf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attn-activation", choices=list(act_class_mapping), default="silu")
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max-z", type=int, default=100)
    parser.add_argument("--max-num-neighbors", type=int, default=32)
    parser.add_argument("--reduce-op", choices=["add", "mean"], default="add")
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--vecnorm-type", default="none")
    parser.add_argument("--trainable-vecnorm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vertex-type", default=None)

    parser.add_argument("--electron-radius", type=float, default=1.0)
    parser.add_argument("--learnable-radius", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-sample-points", type=int, default=6)
    parser.add_argument("--atom-token-extra-neighbors", type=int, default=0)
    parser.add_argument("--electron-grad-penalty", type=float, default=0.0)
    parser.add_argument("--electron-gate", type=float, default=0.15)
    parser.add_argument(
        "--electron-gate-mode",
        choices=["fixed", "schedule", "lr_scale", "plateau", "trainable"],
        default="fixed",
    )
    parser.add_argument("--electron-gate-decay", type=float, default=0.98)
    parser.add_argument("--electron-gate-min", type=float, default=0.0)
    parser.add_argument("--electron-gate-factor", type=float, default=0.5)
    parser.add_argument("--electron-gate-patience", type=int, default=10)
    parser.add_argument("--electron-gate-threshold", type=float, default=0.0)
    parser.add_argument("--electron-gate-threshold-mode", choices=["rel", "abs"], default="rel")
    parser.add_argument("--electron-gate-cooldown", type=int, default=0)
    parser.add_argument("--aeea-five-body-scale", type=float, default=0.03)
    parser.add_argument(
        "--aeea-five-body-use-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ee-cutoff", type=float, default=2.0)
    parser.add_argument("--ee-max-num-neighbors", type=int, default=8)
    parser.add_argument("--ee-scalar-scale", type=float, default=0.0)
    parser.add_argument("--ee-vector-scale", type=float, default=0.0)

    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--precision", type=int, choices=[16, 32], default=32)
    parser.add_argument("--log-dir", default="logs/rmd17")
    parser.add_argument("--task", choices=["train", "inference"], default="train")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--test-interval", type=int, default=3000)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--resume-log-dir", default=None)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--gradient-clip-algorithm", choices=["norm", "value"], default="norm")
    parser.add_argument("--print-test-batches", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-test-limit", type=int, default=-1)
    args = parser.parse_args()
    if not args.dataset_root or not args.dataset_arg:
        parser.error("dataset_root and dataset_arg must be set in the YAML or command line.")
    return args


def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)

    if args.resume_log_dir:
        args.log_dir = args.resume_log_dir
        if args.load_model is None:
            candidate = os.path.join(args.log_dir, "last.ckpt")
            if os.path.exists(candidate):
                args.load_model = candidate

    os.makedirs(args.log_dir, exist_ok=True)
    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    data = EMolDataModule(args)
    data.prepare_dataset()
    model = EMolTask(args, mean=data.mean, std=data.std)

    checkpoint = ModelCheckpoint(
        dirpath=args.log_dir,
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
        every_n_epochs=args.save_interval,
        filename="{epoch}-{val_loss:.4f}",
    )
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        gpus=args.ngpus,
        num_nodes=args.num_nodes,
        accelerator=args.accelerator,
        default_root_dir=args.log_dir,
        callbacks=[
            EarlyStopping("val_loss", patience=args.early_stopping_patience),
            checkpoint,
        ],
        logger=[
            TensorBoardLogger(args.log_dir, name="tensorboard", version=""),
            CSVLogger(args.log_dir, name="", version=""),
        ],
        reload_dataloaders_every_n_epochs=args.reload,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        num_sanity_val_steps=0,
    )

    if args.task == "train":
        trainer.fit(model, datamodule=data, ckpt_path=args.load_model)
        checkpoint_path = checkpoint.best_model_path or os.path.join(args.log_dir, "last.ckpt")
        trainer.test(model=model, ckpt_path=checkpoint_path, datamodule=data)
    else:
        trainer.test(model=model, datamodule=data)

    results = model.inference_results
    energy_mae = np.abs(results["y_true"].numpy() - results["y_pred"].numpy()).mean()
    print(f"Energy MAE: {energy_mae:.6f}")
    if args.derivative:
        force_mae = np.abs(results["dy_true"].numpy() - results["dy_pred"].numpy()).mean()
        print(f"Forces MAE: {force_mae:.6f}")


if __name__ == "__main__":
    main()
