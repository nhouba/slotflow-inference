"""
NPE-per-K baseline that SlotFlow is compared against.

For each cardinality K, a SEPARATE fixed-dimension flow models the joint posterior
over the K *canonically-ordered* components (K*param_dim dims) conditioned on the
set embedding; a classifier on the same embedding selects K. This is "NPE + model
selection": K_max trained models + a separate selection step, O(K_max) to train,
and it must break the permutation symmetry by a canonical ordering (which SlotFlow
avoids via Hungarian matching). The comparison is on quality and cost.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.core import build_conditional_flow, gmm_param_transform, gmm_param_invert
from src.gmm.encoder import DeepSetsEncoder, SetTransformerEncoder


def _canonical_order(p):
    """Sort components (dim 1) by mu_x (coord 0). p: (m, K, param_dim)."""
    order = torch.argsort(p[:, :, 0], dim=1)
    return torch.gather(p, 1, order.unsqueeze(-1).expand(-1, -1, p.size(-1)))


class NPEPerK(nn.Module):
    def __init__(self, k_max=10, param_dim=4, hidden_dim=128, flow_depth=8,
                 in_dim=2, encoder="set_transformer", npe_flow_width=256):
        super().__init__()
        self.k_max = k_max
        self.param_dim = param_dim
        if encoder == "set_transformer":
            self.encoder = SetTransformerEncoder(in_dim=in_dim, hidden_dim=hidden_dim)
        else:
            self.encoder = DeepSetsEncoder(in_dim=in_dim, hidden_dim=hidden_dim)
        ctx = 2 * hidden_dim
        self.classifier = nn.Linear(ctx, k_max)
        # One fixed-dim flow per cardinality (K*param_dim target dims). Use a
        # NARROWER hidden width than SlotFlow's shared flow (default 768): 10
        # separate flows at 768 => 362M params and OOMs on K-diverse batches.
        # 256 keeps NPE ~40M (comparable to SlotFlow's 27.5M) and is plenty for
        # these targets; the baseline's weakness is structural (per-K + canonical
        # ordering), not flow width.
        self.flows = nn.ModuleList([
            build_conditional_flow(param_dim * K, ctx, flow_depth,
                                   hidden_features=npe_flow_width)
            for K in range(1, k_max + 1)
        ])

    def context(self, X):
        _, e_cls = self.encoder(X)
        return e_cls

    def forward(self, X):
        e = self.context(X)
        klog = self.classifier(e)
        return {"e_cls": e, "K_logits": klog,
                "K_pred": torch.argmax(klog, dim=-1) + 1}

    @torch.no_grad()
    def sample(self, X, K, n_samples=500):
        """Posterior samples in NATURAL params for cardinality K. Returns (n, K, param_dim)."""
        e = self.context(X)                                   # (1, ctx)
        z = self.flows[K - 1].sample(n_samples, context=e)    # (1, n, K*param_dim)
        z = z.squeeze(0).view(n_samples, K, self.param_dim)
        return gmm_param_invert(z)


class NPEPerKWrapper(pl.LightningModule):
    def __init__(self, model, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.ce = nn.CrossEntropyLoss()

    def compute_losses(self, X, true_k, params):
        device = self.device
        X, true_k, params = X.to(device), true_k.to(device), params.float().to(device)
        out = self.model(X)
        e = out["e_cls"]
        k_loss = self.ce(out["K_logits"], true_k.long() - 1)

        flow_loss = X.new_tensor(0.0)
        n = 0
        for Kt in torch.unique(true_k):
            K = int(Kt.item())
            idx = true_k == Kt
            m = int(idx.sum().item())
            if m == 0:
                continue
            p = _canonical_order(params[idx][:, :K, :])               # (m, K, P)
            flat = gmm_param_transform(p.reshape(-1, self.model.param_dim))
            target = flat.reshape(m, K * self.model.param_dim)        # (m, K*P)
            ll = self.model.flows[K - 1].log_prob(target, context=e[idx])
            flow_loss = flow_loss + (-ll.mean()) * m
            n += m
        flow_loss = flow_loss / max(n, 1)
        total = k_loss + flow_loss
        return total, {"Total": total, "CE": k_loss, "Flow": flow_loss}

    def training_step(self, batch, _):
        X, K, p = batch
        loss, mtr = self.compute_losses(X, K, p)
        for k, v in mtr.items():
            self.log(f"train/{k}", v.detach().float(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        X, K, p = batch
        loss, mtr = self.compute_losses(X, K, p)
        for k, v in mtr.items():
            self.log(f"val/{k}", v.detach().float(), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=6, min_lr=1e-6)
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val/Total"}
