# ================================================================
# 0. Package import
# ================================================================

import sys
import os
import time
from datetime import datetime, timedelta
import torch
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.strategies import DDPStrategy

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

import torch.distributed as dist

if dist.is_available() and dist.is_initialized():
    print(f"[RANK {dist.get_rank()}] DDP initialized on {torch.cuda.get_device_name()}")
else:
    print("⚠️ DDP not initialized yet")

from src.model import SlotFlow
from src.wrapper import Wrapper
from src.dataset import MultiSinusoidDataset, custom_collate
import argparse

# ================================================================
# 1. Argument parsing
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--max_K", type=int, default=10)
parser.add_argument("--N_train", type=int, default=20000)
parser.add_argument("--out_dir", type=str, default="results")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--flow_depth",
    type=int,
    default=8,
    help="Number of flow coupling layers / depth of the SlotFlow model.",
)
parser.add_argument(
    "--resume_from",
    type=str,
    default=None,
    help="Path to checkpoint for resuming training.",
)
args = parser.parse_args()

# ================================================================
# 2. Setup and logging
# ================================================================
if "LOCAL_RANK" in os.environ:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@rank_zero_only
def safe_print(*a, **kw):
    print(*a, **kw)


class DualLogger:
    """Tee-like logger for both console and file output."""

    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, "a", buffering=1)
        self.encoding = self.terminal.encoding
        self.errors = self.terminal.errors

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


start_time = time.time()
safe_print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Starting script ...")
safe_print(f"Flow depth = {args.flow_depth}")
safe_print("GPUs visible to PyTorch:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    safe_print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# ================================================================
# 3. Dataset
# ================================================================
N = args.N_train
max_components = args.max_K
max_slots = max_components
hidden_dim = 256
tEnd_long, tEnd_short = 300, 10
num_samples_long, num_samples_short = 10 * tEnd_long, 512 * tEnd_short
amp_range, freq_range = (0.5, 1.5), (2.5, 3)
noise_std = (0, 1.5)
use_noise_encoder = False

safe_print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Generating dataset...")
dataset = MultiSinusoidDataset(
    set_size=N,
    num_samples_long=num_samples_long,
    tEnd_long=tEnd_long,
    num_samples_short=num_samples_short,
    tEnd_short=tEnd_short,
    max_components=max_components,
    amp_range=amp_range,
    freq_range=freq_range,
    noise_std=noise_std,
    min_freq_sep=0.01,
)

n_total = len(dataset)
n_train, n_val = int(0.8 * n_total), int(0.2 * n_total)
train_set, val_set = random_split(dataset, [n_train, n_val])

g = torch.Generator().manual_seed(args.seed)
train_loader = DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
    collate_fn=custom_collate,
    num_workers=6,
    pin_memory=True,
    drop_last=True,
    generator=g,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_set,
    batch_size=128,
    shuffle=False,
    collate_fn=custom_collate,
    num_workers=6,
    pin_memory=True,
    persistent_workers=True,
)

# ================================================================
# 4. Output setup
# ================================================================
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(args.out_dir, f"depth_{args.flow_depth}")
ckpt_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

# Redirect stdout to log
sys.stdout = DualLogger(
    os.path.join(output_dir, f"train_log_depth{args.flow_depth}.txt")
)
sys.stderr = sys.stdout

safe_print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Output: {output_dir}")
safe_print(f"Resuming from checkpoint: {args.resume_from}")

# ================================================================
# 5. Model + Lightning setup
# ================================================================
model = SlotFlow(
    hidden_dim=hidden_dim,
    max_slots=max_slots,
    use_noise_encoder=use_noise_encoder,
    flow_depth=args.flow_depth,
)
lit_model = Wrapper(model, lr=1e-4, freq_range=freq_range)

os.makedirs(ckpt_dir, exist_ok=True)
os.chdir(output_dir)


early_stop_callback = EarlyStopping(
    monitor="val/Total", patience=10, verbose=True, mode="min"
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")

checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_dir,
    filename="best",  # always called 'best.ckpt'
    save_top_k=1,  # keep only the single best model
    save_last=True,  # also keep 'last.ckpt' for resume
    monitor="val/Total",
    mode="min",
)

trainer = pl.Trainer(
    max_epochs=300,  # total target epochs; Lightning will stop automatically
    accelerator="gpu",
    devices=4,
    strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=1800)),
    precision="32",
    benchmark=True,
    enable_progress_bar=False,
    deterministic=False,
    log_every_n_steps=10,
    callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
    enable_checkpointing=True,
    default_root_dir=output_dir,
)

# ================================================================
# 6. Train (with resume support)
# ================================================================
safe_print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Starting training ...")

if args.resume_from and os.path.exists(args.resume_from):
    ckpt_path = args.resume_from
    safe_print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
else:
    ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(ckpt_path):
        safe_print(f"[INFO] Found existing last checkpoint: {ckpt_path}")
    else:
        ckpt_path = None
        safe_print("[INFO] No checkpoint found; starting from scratch.")

trainer.fit(lit_model, train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

# ================================================================
# 7. Cleanup: keep only 'best.ckpt' and 'last.ckpt'
# ================================================================
safe_print("[INFO] Cleaning up old checkpoints...")
for f in os.listdir(ckpt_dir):
    if not (f.endswith("best.ckpt") or f.endswith("last.ckpt")):
        os.remove(os.path.join(ckpt_dir, f))
safe_print("[INFO] Checkpoints retained:")
for f in os.listdir(ckpt_dir):
    safe_print("   ", f)
