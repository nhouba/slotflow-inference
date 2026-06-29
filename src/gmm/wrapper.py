"""GMMWrapper -- Lightning training/inference wrapper for the GMM instantiation."""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.core import gmm_param_transform, hungarian_flow_matching_loss


class GMMWrapper(pl.LightningModule):
    def __init__(self, model, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, X, use_gt_k=None):
        return self.model(X, use_gt_k=use_gt_k)

    def compute_losses(self, X, true_k, params):
        device = self.device
        X = X.to(device)
        true_k = true_k.to(device)
        params = params.float().to(device)

        out = self.model(X, use_gt_k=true_k)
        k_loss = self.ce_loss(out["K_logits"], true_k.long() - 1)

        max_slots = params.shape[1]
        mask = torch.arange(max_slots, device=device).unsqueeze(0) < true_k.unsqueeze(1)
        flat = params.view(-1, params.shape[-1])[mask.view(-1)]
        true_param_list = torch.split(flat, true_k.tolist())

        flow_loss = hungarian_flow_matching_loss(
            self.model.flow, out["context"], true_param_list, true_k.long(),
            param_transform=gmm_param_transform)

        total = k_loss + flow_loss
        return total, {"Total": total, "CE": k_loss, "Flow": flow_loss}

    def training_step(self, batch, batch_idx):
        X, K, params = batch
        loss, metrics = self.compute_losses(X, K, params)
        for k, v in metrics.items():
            self.log(f"train/{k}", v.detach().float(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, K, params = batch
        loss, metrics = self.compute_losses(X, K, params)
        for k, v in metrics.items():
            self.log(f"val/{k}", v.detach().float(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=6, min_lr=1e-6)
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val/Total"}
