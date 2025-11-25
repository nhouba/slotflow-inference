import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nflows.flows import Flow
from nflows.transforms import CompositeTransform, ReversePermutation
from nflows.distributions import StandardNormal

from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms import CompositeTransform, ReversePermutation
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)


# ---------------------------
# Reparameterized rsample for Flow
# ---------------------------
def rsample(self, num_samples, context=None):
    """
    Differentiable sampling from flow with context.
    Returns: (num_samples, B, param_dim)
    """
    if context is not None:
        device = context.device
        B, D = context.shape
        context_expanded = context.unsqueeze(0).expand(num_samples, B, D)  # (S,B,D)
        context_flat = context_expanded.reshape(-1, D)  # (S*B,D)
    else:
        raise ValueError("Context required for conditional flow sampling.")

    z = self._distribution.sample(num_samples * B).to(device)  # (S*B, param_dim)
    x, _ = self._transform.inverse(z, context=context_flat)  # (S*B, param_dim)
    return x.view(num_samples, B, -1)


Flow.rsample = rsample


# -------------------------------
# Positional Encoding
# -------------------------------
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# -------------------------------
# Pooling: mean + max + attention
# -------------------------------
class PoolingConcat(nn.Module):
    def __init__(self, d_model, attn_dim=128):
        super().__init__()
        self.attn = nn.Linear(d_model, attn_dim)
        self.context_vec = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, x):
        mean_pool = x.mean(dim=1)  # (B, C)
        max_pool, _ = x.max(dim=1)  # (B, C)

        a = torch.tanh(self.attn(x))  # (B, L, attn_dim)
        scores = self.context_vec(a).squeeze(-1)  # (B, L)
        attn_weights = F.softmax(scores, dim=-1)
        attn_pool = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)

        return torch.cat([mean_pool, max_pool, attn_pool], dim=-1)


# -------------------------------
# SlotFlow
# -------------------------------
class SlotFlow(nn.Module):
    def __init__(
        self, hidden_dim=128, max_slots=3, flow_depth=8, use_noise_encoder=True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_slots = max_slots
        self.use_noise_encoder = use_noise_encoder
        self.flow_depth = flow_depth

        def make_conv_encoder(in_channels=1):
            return nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=8, stride=2, padding=3),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3),
                nn.GELU(),
                nn.Conv1d(
                    64, 128, kernel_size=5, stride=1, padding=3
                ),  # ← was stride=2
                nn.GELU(),
            )

        # --- Attention + positional encodings ---
        # Flow path now uses per-modality attention (long & short)
        self.attn_long_flow = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, batch_first=True
        )
        self.attn_short_flow = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, batch_first=True
        )

        # Classifier keeps long-only attention
        self.global_attn_long = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, batch_first=True
        )

        self.pos = PosEnc(128)

        # --- Encoders ---
        self.encoder_conv_long = make_conv_encoder(
            in_channels=2
        )  # long: freq-domain (Re/Im)
        self.encoder_conv_short = make_conv_encoder(in_channels=1)  # short: time-domain

        # Flow pooling & projection (two pools -> concat -> fc)
        self.pool_flow_long = PoolingConcat(128)  # returns 3*128
        self.pool_flow_short = PoolingConcat(128)  # returns 3*128
        self.encoder_fc = nn.Linear(
            6 * 128, 2 * hidden_dim
        )  # widened from 3*128 to 6*128

        # Optional noise encoder on long freq-domain input
        if self.use_noise_encoder:
            self.encoder_conv_noise = make_conv_encoder(in_channels=2)
            self.pool_noise = PoolingConcat(128)
            self.encoder_fc_noise = nn.Linear(3 * 128, hidden_dim)
            self.noise_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 2),  # outputs [mu, log_std] for log(noise_std)
            )
        else:
            self.encoder_conv_noise = None
            self.pool_noise = None
            self.encoder_fc_noise = None
            self.noise_head = None

        # Classifier projection
        self.pool_long = PoolingConcat(128)
        self.encoder_fc_long = nn.Linear(3 * 128, 2 * hidden_dim)
        self.k_classifier = nn.Linear(2 * hidden_dim, max_slots)

        # --- Conditional flow ---
        param_dim = 4  # [amp, cosφ, sinφ, freq]
        self.flow_context_dim = 2 * hidden_dim + max_slots

        # Stronger but more stable flow
        # n_layers = 8  # number of flow layers
        hidden_features = 768  # width of hidden MLPs
        num_bins = 48  # spline bins

        transforms = []
        for i in range(self.flow_depth):
            transforms.append(ReversePermutation(features=param_dim))
            if i % 2 == 0:
                # Even index → spline layer
                transforms.append(
                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                        features=param_dim,
                        hidden_features=hidden_features,
                        context_features=self.flow_context_dim,
                        num_bins=num_bins,
                        tails="linear",  # stable tails
                        tail_bound=2.0,
                    )
                )
            else:
                # Odd index → affine layer
                transforms.append(
                    MaskedAffineAutoregressiveTransform(
                        features=param_dim,
                        hidden_features=hidden_features,
                        context_features=self.flow_context_dim,
                    )
                )

        # Build flow
        self.flow = Flow(
            transform=CompositeTransform(transforms),
            distribution=StandardNormal([param_dim]),
        )

    def forward(self, x_long, x_short, use_gt_k=None, k_prior=None):
        B, L = x_long.shape

        # --- FFT input (long) as Re/Im channels ---
        fft_long = torch.fft.rfft(x_long, dim=-1, norm="ortho")
        fft_long = torch.view_as_real(fft_long).permute(0, 2, 1)  # (B, 2, Lr)

        # --- Encode long (freq-domain) ---
        h_long = self.encoder_conv_long(fft_long)  # (B, C, L')
        h_long = h_long.permute(0, 2, 1)  # (B, L', C)
        h_long = self.pos(h_long)

        # --- Encode short (time-domain) ---
        h_short = self.encoder_conv_short(x_short.unsqueeze(1))  # (B, C, L'')
        h_short = h_short.permute(0, 2, 1)  # (B, L'', C)
        h_short = self.pos(h_short)

        # =========================
        # FLOW EMBEDDING (per-modality self-attention + fusion)
        # =========================
        h_long_attn_flow, w_long_flow = self.attn_long_flow(
            h_long, h_long, h_long
        )  # (B, L1, 128)
        h_short_attn_flow, w_short_flow = self.attn_short_flow(
            h_short, h_short, h_short
        )  # (B, L2, 128)

        # Pool each modality and fuse
        h_flow_long = self.pool_flow_long(h_long_attn_flow)  # (B, 3*128)
        h_flow_short = self.pool_flow_short(h_short_attn_flow)  # (B, 3*128)
        h_flow_fused = torch.cat([h_flow_long, h_flow_short], dim=-1)  # (B, 6*128)

        # Project to flow embedding
        h_embed_flow = self.encoder_fc(h_flow_fused)  # (B, 2*hidden_dim)

        # =========================
        # CLASSIFIER EMBEDDING (long-only attention)
        # =========================
        h_long_attn_cls, w_long_cls = self.global_attn_long(h_long, h_long, h_long)
        h_long_pool = self.pool_long(h_long_attn_cls)  # (B, 3*128)
        h_embed_long = self.encoder_fc_long(h_long_pool)  # (B, 2*hidden_dim)
        k_logits = self.k_classifier(h_embed_long)

        if k_prior is not None:
            # allow tensor or np array; normalize and move to device if needed
            if not torch.is_tensor(k_prior):
                k_prior = torch.tensor(
                    k_prior, dtype=k_logits.dtype, device=k_logits.device
                )
            else:
                k_prior = k_prior.to(k_logits.device, dtype=k_logits.dtype)
            k_prior = k_prior / (k_prior.sum() + 1e-8)
            log_prior = torch.log(k_prior + 1e-8)
            k_logits = k_logits + log_prior

        k_pred_int = (
            torch.argmax(F.softmax(k_logits, dim=-1), dim=-1) + 1
        )  # in {1..max_slots}
        K = use_gt_k if use_gt_k is not None else k_pred_int

        # =========================
        # NOISE ENCODER (optional)
        # =========================
        if self.use_noise_encoder:
            h_noise = self.encoder_conv_noise(fft_long).permute(0, 2, 1)  # (B, L', C)
            h_noise = self.pos(h_noise)
            h_noise_pool = self.pool_noise(h_noise)  # (B, 3*128)
            h_embed_noise = self.encoder_fc_noise(h_noise_pool)  # (B, hidden_dim)

            log_sigma_params = self.noise_head(h_embed_noise)  # (B, 2)
            mu, log_std = log_sigma_params[:, 0], log_sigma_params[:, 1]
            eps = torch.randn_like(mu)
            log_noise_std_sampled = mu + torch.exp(log_std) * eps
            noise_std_sampled = torch.exp(log_noise_std_sampled)
        else:
            mu = log_std = noise_std_sampled = None

        # =========================
        # FLOW CONTEXT PER SLOT
        # =========================
        per_slot_contexts, batch_slot_ids = [], []
        for b in range(B):
            k_b = int(K[b].item())
            h_b = h_embed_flow[b].expand(k_b, -1)  # (k_b, 2*hidden_dim)
            slot_ids = torch.arange(k_b, device=x_long.device)
            slot_onehot = F.one_hot(
                slot_ids, num_classes=self.max_slots
            )  # (k_b, max_slots)
            ctx_b = torch.cat(
                [h_b, slot_onehot], dim=-1
            )  # (k_b, 2*hidden_dim + max_slots)
            per_slot_contexts.append(ctx_b)
            batch_slot_ids.extend([b] * k_b)

        context_slots = (
            torch.cat(per_slot_contexts, dim=0)
            if len(per_slot_contexts) > 0
            else torch.zeros(
                0, self.flow_context_dim, device=x_long.device, dtype=h_embed_flow.dtype
            )
        )

        return {
            "h_embed": h_embed_flow,
            "K_logits": k_logits,
            "K_pred": K,
            "context": context_slots,
            "batch_slot_ids": batch_slot_ids,
            "noise_std": noise_std_sampled,
            "log_sigma_mu": mu,
            "log_sigma_std": log_std,
            "h_embed_flow": h_embed_flow,
            "h_embed_long": h_embed_long,
            "attn_long_flow": w_long_flow.detach(),
            "attn_short_flow": w_short_flow.detach(),
            "attn_long_cls": w_long_cls.detach(),
        }
