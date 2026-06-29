"""
Evaluation for the Gaussian-mixture benchmark: SlotFlow-GMM vs NPE-per-K vs the
Gibbs reference. Produces the GMM column of the cross-domain comparison:
  - SlotFlow: cardinality accuracy, q(K|x) ECE, SBC uniformity, inference time;
  - NPE-per-K: cardinality accuracy, time, flow-parameter count (the O(K_max) cost);
  - Wasserstein distance (SlotFlow vs the Gibbs reference) per component coordinate,
    at the true K;
  - a BIC reference-cardinality sanity check.

Note: the Gibbs (mu, sigma, w | x, K) sampler is the reference for the per-component
Wasserstein comparison; the BIC-based p(K | x) is an approximate model-order
reference (see baselines/reference_gmm.py).
"""

import argparse

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader

from baselines.npe_per_k import NPEPerK
from baselines.reference_gmm import bic_model_selection, reference_samples
from src.core import gmm_param_invert
from src.eval.metrics import (
    cardinality_accuracy, inference_timing, kpost_ece,
    sbc_ranks, sbc_uniformity_pvalue, wasserstein_per_component,
)
from src.gmm.dataset import GMMDataset
from src.gmm.model import SlotFlowGMM

COORDS = ["mu_x", "mu_y", "log_sx", "log_sy"]


def _relabel_to_truth_persample(samples, true_means):
    """Relabel EACH sample's K components to the true order (Hungarian on mu) --
    this is what fixes Gibbs label-switching. samples (S,K,P); true_means (K,2)."""
    out = samples.copy()
    K = samples.shape[1]
    for t in range(samples.shape[0]):
        r, c = linear_sum_assignment(cdist(samples[t, :, :2], true_means))
        order = np.empty(K, dtype=int)
        order[c] = r
        out[t] = samples[t][order]
    return out


def _relabel_to_truth_mean(samples, true_means):
    """Relabel coherent (non-switching) samples once, by posterior-mean mu."""
    K = samples.shape[1]
    r, c = linear_sum_assignment(cdist(samples.mean(0)[:, :2], true_means))
    order = np.empty(K, dtype=int)
    order[c] = r
    return samples[:, order, :]


def load_lightning(model, path):
    sd = torch.load(path, map_location="cpu").get("state_dict", {})
    msd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(msd, strict=False)
    return model


@torch.no_grad()
def w2_vs_reference(slot_model, dataset, device, n_ref, n_samples=500):
    acc = {c: [] for c in COORDS}
    for idx in range(n_ref):
        X, K, params = dataset[idx]
        true_means = params[:K, :2].numpy()
        out = slot_model(X.unsqueeze(0).to(device),
                         use_gt_k=torch.tensor([K], device=device))
        s = gmm_param_invert(slot_model.flow.rsample(n_samples, context=out["context"]))
        s = s.cpu().numpy()                                  # (n, K, 4)
        ref = reference_samples(X.numpy(), K, seed=idx)      # (S, K, 4), label-switched
        # align BOTH to the true component order (removes Gibbs label-switching)
        s = _relabel_to_truth_mean(s, true_means)
        ref = _relabel_to_truth_persample(ref, true_means)
        for j in range(K):
            ws = wasserstein_per_component(s[:, j, :], ref[:, j, :])
            for c, name in enumerate(COORDS):
                acc[name].append(ws[c])
    return {name: float(np.mean(v)) for name, v in acc.items()}


def bic_cardinality_sanity(dataset, k_max, n_bic):
    hit = 0
    for idx in range(n_bic):
        X, K, _ = dataset[idx]
        pk, _ = bic_model_selection(X.numpy(), range(1, k_max + 1), seed=idx)
        hit += int(max(pk, key=pk.get) == K)
    return hit / max(n_bic, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slotflow_ckpt", required=True)
    p.add_argument("--npe_ckpt", default=None)
    p.add_argument("--k_max", type=int, default=10)
    p.add_argument("--n_points", type=int, default=400)
    # difficulty knobs MUST match the training run so the model is evaluated on the
    # same distribution it was trained on (defaults = hard production benchmark).
    p.add_argument("--sep_min", type=float, default=0.5)
    p.add_argument("--sep_max", type=float, default=3.0)
    p.add_argument("--dirichlet_alpha", type=float, default=1.0)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--encoder", default="set_transformer")
    p.add_argument("--test_size", type=int, default=10000)
    p.add_argument("--n_ref", type=int, default=50)
    p.add_argument("--n_bic", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = GMMDataset(set_size=args.test_size, k_max=args.k_max,
                      n_points=args.n_points, seed=args.seed,
                      sep_range=(args.sep_min, args.sep_max),
                      dirichlet_alpha=args.dirichlet_alpha)
    test_loader = DataLoader(test, batch_size=128, num_workers=4)
    test_bs1 = DataLoader(test, batch_size=1, num_workers=2)

    slot = SlotFlowGMM(hidden_dim=args.hidden_dim, max_slots=args.k_max,
                       param_dim=4, in_dim=2, encoder=args.encoder)
    load_lightning(slot, args.slotflow_ckpt)
    slot = slot.to(dev).eval()

    acc = cardinality_accuracy(slot, test_loader, dev)
    ece, _ = kpost_ece(slot, test_loader, dev)
    ranks = sbc_ranks(slot, test_bs1, dev, invert_fn=gmm_param_invert, n_samples=200, max_n=500)
    sbc_p = {COORDS[c]: round(sbc_uniformity_pvalue(r), 3) for c, r in ranks.items()}
    t_slot = inference_timing(slot, test_bs1, dev, n=200)
    w2 = w2_vs_reference(slot, test, dev, n_ref=args.n_ref)
    bic_acc = bic_cardinality_sanity(test, args.k_max, args.n_bic)

    print("\n===================== GMM BENCHMARK =====================")
    print("[SlotFlow-GMM]")
    print(f"  cardinality accuracy : {acc:.4f}")
    print(f"  q(K|x) ECE           : {ece:.4f}")
    print(f"  SBC uniformity p     : {sbc_p}")
    print(f"  inference time       : {t_slot*1e3:.2f} ms/set")
    print(f"  W2 vs Gibbs ref      : " + ", ".join(f"{k}={v:.3f}" for k, v in w2.items()))
    print(f"[reference] BIC cardinality agreement w/ truth ({args.n_bic} cases): {bic_acc:.3f}")

    if args.npe_ckpt:
        npe = NPEPerK(k_max=args.k_max, param_dim=4, hidden_dim=args.hidden_dim,
                      in_dim=2, encoder=args.encoder)
        load_lightning(npe, args.npe_ckpt)
        npe = npe.to(dev).eval()
        n_correct = n_total = 0
        with torch.no_grad():
            for X, K, _ in test_loader:
                pred = npe(X.to(dev))["K_pred"].cpu()
                n_correct += (pred == K).sum().item()
                n_total += len(K)
        npe_acc = n_correct / max(n_total, 1)
        n_flow_params = sum(q.numel() for f in npe.flows for q in f.parameters())
        slot_flow_params = sum(q.numel() for q in slot.flow.parameters())
        print("[NPE-per-K]")
        print(f"  cardinality accuracy : {npe_acc:.4f}")
        print(f"  flow params          : {n_flow_params/1e6:.1f}M across {args.k_max} flows")
        print(f"  (SlotFlow shared flow: {slot_flow_params/1e6:.1f}M; SlotFlow is O(K_hat), NPE is O(K_max) models)")

    print("NOTE: with BIC as only an approximate p(K|x) reference, the q(K|x)")
    print("      TV-to-reference number is omitted.")
    print("========================================================\n")


if __name__ == "__main__":
    main()
