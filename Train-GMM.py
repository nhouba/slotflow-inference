"""
Training entry point for the Gaussian-mixture benchmark.

Uses on-the-fly infinite data (train seed=None) and builds the GMM instantiation on
the shared core (src/core.py). The NPE-per-K baseline lives in baselines/. This
script trains SlotFlow-GMM and runs the calibration/accuracy half of the evaluation.
"""

import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from src.core import gmm_param_invert
from src.eval.metrics import (
    cardinality_accuracy,
    inference_timing,
    kpost_ece,
    sbc_ranks,
    sbc_uniformity_pvalue,
)
from src.gmm.dataset import GMMDataset
from src.gmm.model import SlotFlowGMM
from src.gmm.wrapper import GMMWrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results_gmm")
    p.add_argument("--k_max", type=int, default=10)
    p.add_argument("--n_points", type=int, default=400)
    # difficulty knobs (defaults = the hard production benchmark). The easier
    # regime widens separation and balances cluster sizes to show a high-accuracy
    # cross-domain point (proving generality is benchmark-limited, not method-limited).
    p.add_argument("--sep_min", type=float, default=0.5)
    p.add_argument("--sep_max", type=float, default=3.0)
    p.add_argument("--dirichlet_alpha", type=float, default=1.0)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--flow_depth", type=int, default=8)
    p.add_argument("--encoder", default="set_transformer")  # or "deep_sets"
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--steps_per_epoch", type=int, default=4000)
    p.add_argument("--val_size", type=int, default=10000)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)

    common = dict(k_max=args.k_max, n_points=args.n_points,
                  sep_range=(args.sep_min, args.sep_max),
                  dirichlet_alpha=args.dirichlet_alpha)
    # on-the-fly infinite data for training (seed=None => fresh each step)
    train_set = GMMDataset(set_size=args.steps_per_epoch * args.batch_size,
                           seed=None, **common)
    val_set = GMMDataset(set_size=args.val_size, seed=args.seed, **common)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=6, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=6,
                            persistent_workers=True)

    model = SlotFlowGMM(hidden_dim=args.hidden_dim, max_slots=args.k_max,
                        flow_depth=args.flow_depth, param_dim=4, in_dim=2,
                        encoder=args.encoder)
    lit = GMMWrapper(model, lr=args.lr)

    ckpt = ModelCheckpoint(dirpath=f"{args.out_dir}/checkpoints", monitor="val/Total",
                           save_top_k=1, save_last=True, mode="min")
    early = EarlyStopping(monitor="val/Total", patience=12, mode="min")
    logger = CSVLogger(args.out_dir, name="gmm")

    # SlotFlow's shared flow/classifier/encoder are always used -> no unused params
    strategy = DDPStrategy(find_unused_parameters=False) if args.devices > 1 else "auto"
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=strategy,
        precision="32",
        gradient_clip_val=5.0,   # flow NLL can spike -> clip to prevent divergence
        callbacks=[ckpt, early],
        logger=logger,
    )
    trainer.fit(lit, train_loader, val_loader)

    if not trainer.is_global_zero:   # rank-0 only for the post-train eval/print
        return

    # ---- accuracy + calibration half of the eval harness ----
    dev = lit.device
    model = model.to(dev).eval()
    val_bs1 = DataLoader(val_set, batch_size=1, num_workers=2)

    acc = cardinality_accuracy(model, val_loader, dev)
    ece, _ = kpost_ece(model, val_loader, dev)
    ranks = sbc_ranks(model, val_bs1, dev, invert_fn=gmm_param_invert, n_samples=200, max_n=500)
    coord_names = ["mu_x", "mu_y", "log_sx", "log_sy"]
    sbc_p = {coord_names[c]: round(sbc_uniformity_pvalue(r), 3) for c, r in ranks.items()}
    t_med = inference_timing(model, val_bs1, dev, n=200)

    print("\n================ GMM BENCHMARK (accuracy + calibration) ================")
    print(f"cardinality accuracy : {acc:.4f}")
    print(f"q(K|x) ECE           : {ece:.4f}   (lower is better)")
    print(f"SBC uniformity p-vals: {sbc_p}   (>0.05 => consistent with calibration)")
    print(f"median inference time: {t_med*1e3:.2f} ms/set")
    print("NOTE: posterior-fidelity (Wasserstein vs the Gibbs reference) is computed")
    print("      by Eval-GMM.py; see baselines/reference_gmm.py.")
    print("========================================================================\n")


if __name__ == "__main__":
    main()
