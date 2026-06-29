"""
Consolidated SlotFlow framework core.

This is the shared, modality-agnostic core used by both instantiations:
    - build_conditional_flow
    - build_slot_contexts
    - rsample
    - hungarian_flow_matching_loss + param transforms

The bodies are identical (modulo self.<attr> -> argument) to the sinusoid model in
src/model.py; the GMM instantiation imports them from here.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, ReversePermutation
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)


# ----------------------------------------------------------------------
# Differentiable sampling (monkey-patched onto Flow), from model.py:18-36
# ----------------------------------------------------------------------
def rsample(self, num_samples, context=None):
    """Differentiable sampling from a conditional flow. Returns (num_samples, B, param_dim)."""
    if context is not None:
        device = context.device
        B, D = context.shape
        context_expanded = context.unsqueeze(0).expand(num_samples, B, D)
        context_flat = context_expanded.reshape(-1, D)
    else:
        raise ValueError("Context required for conditional flow sampling.")
    z = self._distribution.sample(num_samples * B).to(device)
    x, _ = self._transform.inverse(z, context=context_flat)
    return x.view(num_samples, B, -1)


Flow.rsample = rsample


# ----------------------------------------------------------------------
# Model-side core (from model.py:152-190 and 272-292)
# ----------------------------------------------------------------------
def build_conditional_flow(param_dim, context_dim, flow_depth,
                           hidden_features=768, num_bins=48, tail_bound=2.0):
    """Masked-autoregressive RQ-spline + affine flow, conditioned on `context_dim`."""
    transforms = []
    for i in range(flow_depth):
        transforms.append(ReversePermutation(features=param_dim))
        if i % 2 == 0:
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=param_dim, hidden_features=hidden_features,
                context_features=context_dim, num_bins=num_bins,
                tails="linear", tail_bound=tail_bound))
        else:
            transforms.append(MaskedAffineAutoregressiveTransform(
                features=param_dim, hidden_features=hidden_features,
                context_features=context_dim))
    return Flow(transform=CompositeTransform(transforms),
                distribution=StandardNormal([param_dim]))


def build_slot_contexts(h_embed_flow, K, max_slots, device):
    """Per-slot context c_k = [g, one_hot(k)] for k=1..K_b, concatenated over the batch."""
    B = h_embed_flow.shape[0]
    per_slot, batch_slot_ids = [], []
    for b in range(B):
        k_b = int(K[b].item())
        h_b = h_embed_flow[b].expand(k_b, -1)
        slot_ids = torch.arange(k_b, device=device)
        slot_onehot = F.one_hot(slot_ids, num_classes=max_slots)
        per_slot.append(torch.cat([h_b, slot_onehot], dim=-1))
        batch_slot_ids.extend([b] * k_b)
    ctx_dim = h_embed_flow.shape[1] + max_slots
    context = (torch.cat(per_slot, dim=0) if per_slot
               else torch.zeros(0, ctx_dim, device=device, dtype=h_embed_flow.dtype))
    return context, batch_slot_ids


# ----------------------------------------------------------------------
# Permutation-invariant matched loss + instantiation-specific transforms
# ----------------------------------------------------------------------
def hungarian_flow_matching_loss(flow, context, true_params, true_ks,
                                 device="cpu", param_transform=None):
    """Hungarian-matched NLL. `param_transform` (optional) maps component params into
    the flow's modeling coordinates before evaluation; the matching+NLL is generic."""
    B = len(true_ks)
    all_ctx_flat, all_gt_flat, batch_info = [], [], []
    slot_ptr = 0
    for b in range(B):
        k_b = true_ks[b].item()
        gt = true_params[b]
        gt = gt if param_transform is None else param_transform(gt)
        ctx = context[slot_ptr:slot_ptr + k_b]
        ctx_exp = ctx.unsqueeze(0).expand(k_b, -1, -1)
        gt_exp = gt.unsqueeze(1).expand(-1, k_b, -1)
        all_ctx_flat.append(ctx_exp.reshape(-1, ctx.shape[-1]))
        all_gt_flat.append(gt_exp.reshape(-1, gt.shape[-1]))
        batch_info.append((b, k_b))
        slot_ptr += k_b
    all_ctx = torch.cat(all_ctx_flat, dim=0)
    all_gt = torch.cat(all_gt_flat, dim=0)
    all_logps = flow.log_prob(all_gt, context=all_ctx)
    weighted_sum, total_slots, logp_ptr = 0.0, 0, 0
    for b, k_b in batch_info:
        num_pairs = k_b * k_b
        cost = -all_logps[logp_ptr:logp_ptr + num_pairs].view(k_b, k_b)
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        weighted_sum += cost[row, col].mean() * k_b
        total_slots += k_b
        logp_ptr += num_pairs
    return weighted_sum / total_slots


def sinusoid_param_transform(gt, freq_range=(2.5, 3.0), phase_weight=2.0, freq_weight=3.0):
    """Sinusoid instantiation transform (phase->cos/sin already applied upstream)."""
    gt = gt.clone()
    gt[:, 3] = gt[:, 3] - (freq_range[0] + freq_range[1]) / 2
    gt[:, 1:3] *= phase_weight
    gt[:, 3] *= freq_weight
    return gt


# GMM instantiation transform: param order [mu_x, mu_y, log s_x, log s_y]
# (param_dim=4; weights are generative-only, not a flow target -- dropping them
# removes the heavy-tailed logit coordinate that destabilized training and broke
# SBC). Standardizes each coordinate into the spline support (tail_bound=2.0).
_GMM_T = dict(mean_scale=3.0, logscale_center=-0.6, logscale_scale=0.6)


def gmm_param_transform(gt, **kw):
    t = {**_GMM_T, **kw}
    gt = gt.clone()
    gt[:, 0:2] = gt[:, 0:2] / t["mean_scale"]
    gt[:, 2:4] = (gt[:, 2:4] - t["logscale_center"]) / t["logscale_scale"]
    return gt


def gmm_param_invert(s, **kw):
    """Inverse of gmm_param_transform (flow space -> natural params), for evaluation."""
    t = {**_GMM_T, **kw}
    out = s.clone()
    out[..., 0:2] = out[..., 0:2] * t["mean_scale"]
    out[..., 2:4] = out[..., 2:4] * t["logscale_scale"] + t["logscale_center"]
    return out
