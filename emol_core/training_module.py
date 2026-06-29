import torch
from pytorch_lightning import LightningModule
from torch.nn.functional import l1_loss, mse_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import csv
import os

from emol_core.model.factory import create_model, load_model


class EMolTask(LightningModule):
    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super().__init__()

        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        else:
            self.model = create_model(self.hparams, mean, std)

        self._reset_losses_dict()
        self._reset_ema_dict()
        self._reset_inference_results()
        self._reset_electron_gate_plateau_state()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.lr_min,
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        loss_fn = mse_loss if self.hparams.loss_type == 'MSE' else l1_loss
        
        return self.step(batch, loss_fn, "train", batch_idx=batch_idx)

    def on_train_epoch_start(self):
        self._maybe_update_electron_gate()

    def validation_step(self, batch, batch_idx, *args):
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            # validation step
            return self.step(batch, mse_loss, "val", batch_idx=batch_idx)
        # test step
        return self.step(batch, l1_loss, "test", batch_idx=batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "test", batch_idx=batch_idx)

    def step(self, batch, loss_fn, stage, batch_idx=None):
        # Optional: penalize sensitivity to electron coordinates to reduce gradient noise
        electron_grad_penalty = getattr(self.hparams, "electron_grad_penalty", 0.0)
        if stage == "train" and electron_grad_penalty > 0 and hasattr(batch, "elec_coords"):
            batch.elec_coords = batch.elec_coords.detach().requires_grad_(True)

        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            pred, deriv = self(batch)
            
        if stage == "test":
            emae = np.abs((pred - batch.y).detach().cpu().numpy()).mean()
            fmae = np.abs((deriv - batch.dy).detach().cpu().numpy()).mean() if self.hparams.derivative else 0.0
            self._maybe_print_test_batch(batch_idx, pred, batch.y, deriv, batch.dy, emae, fmae)
            if not np.isnan(emae) and not np.isnan(fmae):
                self.inference_results['y_pred'].append(pred.squeeze(-1).detach().cpu())
                self.inference_results['y_true'].append(batch.y.squeeze(-1).detach().cpu())
                if self.hparams.derivative:
                    self.inference_results['dy_pred'].append(deriv.squeeze(-1).detach().cpu())
                    self.inference_results['dy_true'].append(batch.dy.squeeze(-1).detach().cpu())
            else:
                print(f"Skipping batch with NaN: energy_mae={emae}, force_mae={fmae}")

        loss_y, loss_dy = 0, 0
        if self.hparams.derivative:
            if "y" not in batch:
                deriv = deriv + pred.sum() * 0

            loss_dy = loss_fn(deriv, batch.dy)
            
            if stage in ["train", "val"] and self.hparams.loss_scale_dy < 1:
                if self.ema[stage + "_dy"] is None:
                    self.ema[stage + "_dy"] = loss_dy.detach()
                # apply exponential smoothing over batches to dy
                loss_dy = (
                    self.hparams.loss_scale_dy * loss_dy
                    + (1 - self.hparams.loss_scale_dy) * self.ema[stage + "_dy"]
                )
                self.ema[stage + "_dy"] = loss_dy.detach()

            if self.hparams.force_weight > 0:
                self.losses[stage + "_dy"].append(loss_dy.detach())

        if "y" in batch:
            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)

            loss_y = loss_fn(pred, batch.y)
            
            if stage in ["train", "val"] and self.hparams.loss_scale_y < 1:
                if self.ema[stage + "_y"] is None:
                    self.ema[stage + "_y"] = loss_y.detach()
                # apply exponential smoothing over batches to y
                loss_y = (
                    self.hparams.loss_scale_y * loss_y
                    + (1 - self.hparams.loss_scale_y) * self.ema[stage + "_y"]
                )
                self.ema[stage + "_y"] = loss_y.detach()
            
            if self.hparams.energy_weight > 0:
                self.losses[stage + "_y"].append(loss_y.detach())

        loss = loss_y * self.hparams.energy_weight + loss_dy * self.hparams.force_weight

        # Add gradient penalty on electron coordinates to damp noisy electron pathways
        if stage == "train" and electron_grad_penalty > 0 and hasattr(batch, "elec_coords") and batch.elec_coords.requires_grad:
            grad_elec = torch.autograd.grad(
                pred.sum(),
                batch.elec_coords,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad_elec is not None:
                grad_penalty = (grad_elec.pow(2).sum(dim=-1)).mean()
                loss = loss + electron_grad_penalty * grad_penalty
        
        self.losses[stage].append(loss.detach())
        
        return loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.hparams.lr_warmup_steps))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def training_epoch_end(self, training_step_outputs):
        dm = self.trainer.datamodule
        if hasattr(dm, "test_dataset") and len(dm.test_dataset) > 0:
            delta = 0 if self.hparams.reload == 1 else 1
            should_reset = (
                (self.current_epoch + delta + 1) % self.hparams.test_interval == 0
                or ((self.current_epoch + delta) % self.hparams.test_interval == 0 and self.current_epoch != 0)
            )
            if should_reset:
                self.trainer.reset_val_dataloader()
                self.trainer.fit_loop.epoch_loop.val_loop.epoch_loop._reset_dl_batch_idx(len(self.trainer.val_dataloaders))

    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.sanity_checking:
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                # "radius": self.model.representation_model.embedding.electron_features.radius,
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }
            self._maybe_update_electron_gate_on_plateau(result_dict["val_loss"])
            gate_value = self._get_electron_gate_value()
            if gate_value is not None:
                result_dict["electron_gate"] = gate_value

            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

            # if prediction and derivative are present, also log them separately
            if len(self.losses["train_y"]) > 0 and len(self.losses["train_dy"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
                result_dict["train_loss_dy"] = torch.stack(self.losses["train_dy"]).mean()
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
                result_dict["val_loss_dy"] = torch.stack(self.losses["val_dy"]).mean()

            if len(self.losses["test_y"]) > 0 and len(self.losses["test_dy"]) > 0:
                result_dict["test_loss_y"] = torch.stack(self.losses["test_y"]).mean()
                result_dict["test_loss_dy"] = torch.stack(self.losses["test_dy"]).mean()

            self.log_dict(result_dict, sync_dist=True)
            self._log_ratio_metrics_csv()
            
        self._reset_losses_dict()
        self._reset_inference_results()

    def _log_ratio_metrics_csv(self):
        if not getattr(self.trainer, "is_global_zero", True):
            return

        representation = getattr(self.model, "representation_model", None)
        if representation is None or not hasattr(representation, "hybrid_mp_layers"):
            return

        layers = getattr(representation, "hybrid_mp_layers", [])
        rows = []
        epoch_value = int(self.current_epoch)
        for layer_idx, layer in enumerate(layers):
            for ratio_name in ("ae_ratio", "ea_ratio"):
                if not hasattr(layer, ratio_name):
                    continue
                ratio_tensor = getattr(layer, ratio_name).detach().cpu().view(-1)
                if ratio_tensor.numel() != 6:
                    continue
                rows.append({
                    "epoch": epoch_value,
                    "layer": layer_idx,
                    "ratio_type": ratio_name,
                    "s1": float(ratio_tensor[0].item()),
                    "s2": float(ratio_tensor[1].item()),
                    "s3": float(ratio_tensor[2].item()),
                    "s4": float(ratio_tensor[3].item()),
                    "s5": float(ratio_tensor[4].item()),
                    "s6": float(ratio_tensor[5].item()),
                })

        if len(rows) == 0:
            return

        log_dir = getattr(self.hparams, "log_dir", None)
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        ratio_csv_path = os.path.join(log_dir, "ratio_metrics.csv")
        file_exists = os.path.exists(ratio_csv_path)
        fieldnames = ["epoch", "layer", "ratio_type", "s1", "s2", "s3", "s4", "s5", "s6"]

        with open(ratio_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def test_epoch_end(self, outputs) -> None:
        for key in self.inference_results.keys():
            if len(self.inference_results[key]) > 0:
                self.inference_results[key] = torch.cat(self.inference_results[key], dim=0)

    def _maybe_print_test_batch(self, batch_idx, pred, y_true, deriv, dy_true, emae, fmae):
        if not getattr(self.hparams, "print_test_batches", False):
            return

        limit = int(getattr(self.hparams, "print_test_limit", -1))
        if limit >= 0 and batch_idx is not None and batch_idx >= limit:
            return

        pred_cpu = pred.detach().cpu()
        y_true_cpu = y_true.detach().cpu()
        print(f"[test batch {batch_idx}] emae={emae:.6f} y_pred={pred_cpu} y_true={y_true_cpu}")

        if self.hparams.derivative and deriv is not None and dy_true is not None:
            deriv_cpu = deriv.detach().cpu()
            dy_true_cpu = dy_true.detach().cpu()
            print(
                f"[test batch {batch_idx}] fmae={fmae:.6f} "
                f"dy_pred_shape={tuple(deriv_cpu.shape)} dy_true_shape={tuple(dy_true_cpu.shape)} "
                f"dy_pred_has_nan={torch.isnan(deriv_cpu).any().item()} dy_true_has_nan={torch.isnan(dy_true_cpu).any().item()}"
            )

    def _reset_losses_dict(self):
        self.losses = {
            "train": [], "val": [], "test": [],
            "train_y": [], "val_y": [], "test_y": [],
            "train_dy": [], "val_dy": [], "test_dy": [],
        }

    def _reset_inference_results(self):
        self.inference_results = {'y_pred': [], 'y_true': [], 'dy_pred': [], 'dy_true': []}
        
    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}

    def _reset_electron_gate_plateau_state(self):
        self._gate_plateau_best = None
        self._gate_plateau_num_bad_epochs = 0
        self._gate_plateau_cooldown_counter = 0

    def _maybe_update_electron_gate(self):
        if getattr(self.hparams, "model", None) not in {"EMolRepresentation", "RAGEDSampledBlock"}:
            return
        mode = getattr(self.hparams, "electron_gate_mode", "fixed")
        if mode not in {"schedule", "lr_scale"}:
            return
        init_gate = getattr(self.hparams, "electron_gate", 0.25)
        min_gate = getattr(self.hparams, "electron_gate_min", 0.0)
        if mode == "schedule":
            decay = getattr(self.hparams, "electron_gate_decay", 1.0)
            gate = max(min_gate, float(init_gate) * (float(decay) ** float(self.current_epoch)))
        else:
            if not getattr(self.trainer, "optimizers", None):
                return
            base_lr = float(getattr(self.hparams, "lr", 0.0))
            current_lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
            lr_scale = current_lr / base_lr if base_lr > 0 else 1.0
            gate = max(min_gate, float(init_gate) * lr_scale)
        representation = getattr(self.model, "representation_model", None)
        if representation is not None and hasattr(representation, "set_electron_gate"):
            representation.set_electron_gate(gate)

    def _maybe_update_electron_gate_on_plateau(self, val_loss):
        if getattr(self.hparams, "model", None) not in {"EMolRepresentation", "RAGEDSampledBlock"}:
            return
        mode = getattr(self.hparams, "electron_gate_mode", "fixed")
        if mode != "plateau":
            return
        if val_loss is None:
            return
        val_loss = float(val_loss.detach().cpu().item())
        threshold = float(getattr(self.hparams, "electron_gate_threshold", 0.0))
        threshold_mode = getattr(self.hparams, "electron_gate_threshold_mode", "rel")
        patience = int(getattr(self.hparams, "electron_gate_patience", 10))
        cooldown = int(getattr(self.hparams, "electron_gate_cooldown", 0))
        factor = float(getattr(self.hparams, "electron_gate_factor", 0.5))
        min_gate = float(getattr(self.hparams, "electron_gate_min", 0.0))

        def is_improvement(current, best):
            if threshold_mode == "abs":
                return current < best - threshold
            return current < best * (1.0 - threshold)

        if self._gate_plateau_best is None:
            self._gate_plateau_best = val_loss
            return

        if is_improvement(val_loss, self._gate_plateau_best):
            self._gate_plateau_best = val_loss
            self._gate_plateau_num_bad_epochs = 0
            return

        if self._gate_plateau_cooldown_counter > 0:
            self._gate_plateau_cooldown_counter -= 1
            self._gate_plateau_num_bad_epochs = 0
            return

        self._gate_plateau_num_bad_epochs += 1
        if self._gate_plateau_num_bad_epochs <= patience:
            return

        representation = getattr(self.model, "representation_model", None)
        if representation is None or not hasattr(representation, "set_electron_gate"):
            return
        current_gate = self._get_electron_gate_value()
        if current_gate is None:
            return
        new_gate = max(min_gate, float(current_gate) * float(factor))
        representation.set_electron_gate(new_gate)
        self._gate_plateau_num_bad_epochs = 0
        self._gate_plateau_cooldown_counter = cooldown

    def _get_electron_gate_value(self):
        if getattr(self.hparams, "model", None) not in {"EMolRepresentation", "RAGEDSampledBlock"}:
            return None
        representation = getattr(self.model, "representation_model", None)
        if representation is None:
            return None
        if hasattr(representation, "get_electron_gate_value"):
            return representation.get_electron_gate_value()
        return None
