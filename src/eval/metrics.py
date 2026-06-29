"""
Domain-agnostic evaluation metrics.

Designed so the same functions run for the sinusoid and GMM instantiations, giving
the cross-domain comparison. The model-specific bits enter only through:
  - `model(X, use_gt_k=...)` returning {"K_logits","K_pred","context"},
  - `model.flow.rsample(n, context)` (flow-space samples),
  - `invert_fn` mapping flow-space samples -> natural parameters (e.g.
    src.core.gmm_param_invert).

Metric suite:
  cardinality_accuracy · kpost_ece (q(K|x) calibration) · sbc_ranks ·
  wasserstein_per_component (vs a reference) · inference_timing.
"""

import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

try:
    from scipy.stats import wasserstein_distance
except Exception:  # pragma: no cover
    wasserstein_distance = None


@torch.no_grad()
def cardinality_accuracy(model, loader, device):
    """Argmax q(K|x) accuracy vs true K (predicted K, use_gt_k=None)."""
    correct = total = 0
    for X, K, _ in loader:
        pred = model(X.to(device))["K_pred"].cpu()
        correct += (pred == K).sum().item()
        total += len(K)
    return correct / max(total, 1)


@torch.no_grad()
def kpost_ece(model, loader, device, n_bins=15):
    """Expected calibration error of the model-order posterior q(K|x).

    Confidence = max softmax prob; accuracy = (argmax == true K). Returns (ece,
    reliability) where reliability is a list of (mean_conf, acc, frac) per bin.
    """
    confs, hits = [], []
    for X, K, _ in loader:
        logits = model(X.to(device))["K_logits"]
        p = torch.softmax(logits, dim=-1).cpu()
        conf, pred = p.max(dim=-1)
        confs.append(conf)
        hits.append((pred + 1 == K).float())
    conf = torch.cat(confs).numpy()
    hit = torch.cat(hits).numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    ece, reliability = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, c, frac = hit[m].mean(), conf[m].mean(), m.mean()
        ece += abs(acc - c) * frac
        reliability.append((float(c), float(acc), float(frac)))
    return float(ece), reliability


@torch.no_grad()
def sbc_ranks(model, loader_bs1, device, invert_fn, n_samples=200, max_n=1000):
    """Simulation-based calibration ranks per parameter coordinate.

    For each set: condition on the TRUE K, draw posterior samples, invert to natural
    params, Hungarian-match predicted means to the truth, and record per-coordinate
    rank = #(samples < true). Calibrated <=> ranks are Uniform{0..n_samples}.
    Returns {coord_index: np.ndarray of ranks}.
    """
    ranks = None
    n_sets = 0
    for X, K, params in loader_bs1:
        k = int(K.item())
        out = model(X.to(device), use_gt_k=K.to(device))
        s = model.flow.rsample(n_samples, context=out["context"])  # (S, k, P)
        s = invert_fn(s).cpu()
        true = params[0][:k]                                       # (k, P)
        P = true.shape[-1]
        if ranks is None:
            ranks = {c: [] for c in range(P)}
        mu_pred = s.mean(0)[:, :2]                                  # match on means
        row, col = linear_sum_assignment(torch.cdist(mu_pred, true[:, :2]).numpy())
        for si, tj in zip(row, col):
            for c in range(P):
                ranks[c].append(int((s[:, si, c] < true[tj, c]).sum().item()))
        n_sets += 1
        if n_sets >= max_n:
            break
    return {c: np.asarray(v) for c, v in ranks.items()} if ranks else {}


def sbc_uniformity_pvalue(rank_array, n_samples=200):
    """Chi-square p-value for Uniform{0..n_samples} of an SBC rank array.

    p > 0.05 => not rejected (consistent with calibration). Self-contained
    (no scipy.stats dependency)."""
    from math import erfc, sqrt
    obs, _ = np.histogram(rank_array, bins=min(20, n_samples + 1),
                          range=(0, n_samples + 1))
    exp = obs.sum() / len(obs)
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    dof = len(obs) - 1
    # Wilson-Hilferty normal approximation to the chi-square upper tail
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / sqrt(2 / (9 * dof))
    return 0.5 * erfc(z / sqrt(2))


def wasserstein_per_component(model_samples, ref_samples):
    """W2 per coordinate between matched-component samples and a reference.

    model_samples, ref_samples: (S, P) arrays of natural params for ONE matched
    component (matching/iteration is the caller's job; the reference comes from
    baselines/reference_gmm.py). Returns a length-P list of Wasserstein distances.
    """
    if wasserstein_distance is None:
        raise RuntimeError("scipy.stats.wasserstein_distance unavailable")
    P = model_samples.shape[-1]
    return [float(wasserstein_distance(model_samples[:, c], ref_samples[:, c]))
            for c in range(P)]


@torch.no_grad()
def inference_timing(model, loader, device, n=200):
    """Median per-set inference wall-clock (seconds). Use a batch_size=1 loader."""
    times = []
    for i, (X, _, _) in enumerate(loader):
        X = X.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(X)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if i + 1 >= n:
            break
    return float(np.median(times))
