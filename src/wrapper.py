from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from src.loss import hungarian_flow_matching_loss, noise_supervision_loss


@rank_zero_only
def safe_print(*args, **kwargs):
    print(*args, **kwargs)


class Wrapper(pl.LightningModule):
    """
    LightningModule wrapper for a model that predicts parameters using a mixture of classification
    and flow-based regression. Supports additional noise supervision if the model includes a noise encoder.

    Parameters
    ----------
    model : nn.Module
        The underlying PyTorch model to be trained.
    lr : float, optional
        Initial learning rate for the optimizer (default is 1e-3).

    Attributes
    ----------
    model : nn.Module
        The underlying model.
    lr : float
        Learning rate.
    ce_loss : nn.CrossEntropyLoss
        Cross-entropy loss function used for predicting the number of components.
    """

    def __init__(
        self,
        model: nn.Module,
        lr=1e-3,
        phase_weight=2.0,
        freq_weight=3.0,
        freq_range=(2.5, 3.0),
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.phase_weight = phase_weight
        self.freq_weight = freq_weight
        self.freq_range = freq_range
        self.ce_loss = nn.CrossEntropyLoss()  # for predicting number of components

    def forward(self, x_long, x_short, t_full=None, use_gt_k=None):
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        t_full : optional
            Unused placeholder for time-dependent models.
        use_gt_k : torch.Tensor, optional
            True number of components (used for teacher forcing or masking in some models).

        Returns
        -------
        dict
            Output dictionary from the model, typically containing:
            - "K_logits": logits for component classification
            - "context": context embeddings for flow-based regression
            - "noise_std": predicted noise standard deviations (optional)
        """
        return self.model(x_long, x_short, use_gt_k=use_gt_k)

    def compute_losses(self, x_long, x_short, true_k, params, noise_stds):
        device = self.device
        x_long = x_long.to(device)
        x_short = x_short.to(device)
        true_k = true_k.to(device)
        params = params.float()
        noise_stds = noise_stds.to(device)

        B = x_long.shape[0]
        true_k_idx = true_k.long() - 1

        # forward pass
        out = self(x_long, x_short, use_gt_k=true_k)

        # classification loss
        k_loss = self.ce_loss(out["K_logits"], true_k_idx)

        # mask for active components
        max_slots = params.shape[1]
        mask = torch.arange(max_slots, device=device).unsqueeze(0) < true_k.unsqueeze(1)

        # convert phase to (cos, sin)
        phase = params[..., 1]
        cos_phi, sin_phi = torch.cos(phase), torch.sin(phase)
        params_mod = torch.stack(
            [params[..., 0], cos_phi, sin_phi, params[..., 2]], dim=-1
        )

        # flatten and split by batch
        flat_params = params_mod.view(-1, params_mod.shape[-1])
        flat_mask = mask.view(-1)
        true_param_list = torch.split(flat_params[flat_mask], true_k.tolist())

        # flow loss
        flow_loss = hungarian_flow_matching_loss(
            self.model.flow,
            out["context"],
            true_param_list,
            true_k.long(),
            phase_weight=self.phase_weight,
            freq_weight=self.freq_weight,
            freq_range=self.freq_range,
        )

        # noise loss
        if (
            self.model.use_noise_encoder
            and out.get("noise_std") is not None
            and noise_stds.max() > 0
        ):
            noise_loss = noise_supervision_loss(out["noise_std"], noise_stds)
        else:
            noise_loss = torch.tensor(0.0, device=device)

        total_loss = k_loss + flow_loss + 10.0 * noise_loss

        metrics = {
            "Total": total_loss,
            "CE": k_loss,
            "Flow": flow_loss,
            "Noise": noise_loss,
        }
        return total_loss, metrics

    def training_step(self, batch, batch_idx):
        x_long, x_short, true_k, comps, params, noise_stds = batch
        params = params.float().to(self.device)
        true_k = true_k.to(self.device)
        noise_stds = noise_stds.to(self.device)

        loss, metrics = self.compute_losses(x_long, x_short, true_k, params, noise_stds)

        for key, value in metrics.items():
            self.log(
                f"train/{key}",
                value.detach().float(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss

        # Log metrics for monitoring
        for key, value in metrics.items():
            self.log(
                f"train/{key}" if self.training else f"val/{key}",
                value.detach().float(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x_long, x_short, true_k, comps, params, noise_stds = batch
        params = params.float().to(self.device)
        true_k = true_k.to(self.device)
        noise_stds = noise_stds.to(self.device)

        val_loss, metrics = self.compute_losses(
            x_long, x_short, true_k, params, noise_stds
        )

        for key, value in metrics.items():
            self.log(
                f"val/{key}",
                value.detach().float(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return val_loss

    def on_train_epoch_end(self):
        """
        At the end of each training epoch, print nicely formatted metrics and learning rate.
        """
        if self.trainer.is_last_batch:  # only once per epoch
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

            # Extract and clean metrics
            train_metrics = {
                k.replace("train/", ""): v
                for k, v in self.trainer.callback_metrics.items()
                if k.startswith("train/")
            }
            val_metrics = {
                k.replace("val/", ""): v
                for k, v in self.trainer.callback_metrics.items()
                if k.startswith("val/")
            }

            # Convert tensors to floats
            train_metrics = {
                k: (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in train_metrics.items()
            }
            val_metrics = {
                k: (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in val_metrics.items()
            }

            # Format metrics for printing
            train_str = ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            val_str = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())

            # Get current learning rate from optimizer
            if self.trainer.optimizers:
                lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            else:
                lr = 0.0

            safe_print(
                f"[{timestamp}] Epoch {self.current_epoch} | Train: {train_str} | Val: {val_str} | LR: {lr:.2e}"
            )

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns
        -------
        dict
            Dictionary containing optimizer, scheduler, and metric to monitor.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/Total",
        }
