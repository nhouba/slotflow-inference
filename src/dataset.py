import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import math
import random


def custom_collate(batch):
    """
    Collate function for MultiSinusoidDataset.
    Supports both modes:
      - 6-item tuples (train mode, no master)
      - 7-item tuples (debug or inference, with master)
    """
    sample_len = len(batch[0])

    if sample_len == 6:
        xs_long, xs_short, true_ks, comps, params, noise_stds = zip(*batch)

        xs_long = torch.stack(xs_long)
        xs_short = torch.stack(xs_short)
        true_ks = torch.tensor(true_ks, dtype=torch.long)
        comps = torch.stack(comps)
        params = torch.stack(params)
        noise_stds = torch.tensor(noise_stds, dtype=torch.float32)

        return xs_long, xs_short, true_ks, comps, params, noise_stds

    elif sample_len == 7:
        xs_long, xs_short, true_ks, comps, params, noise_stds, x_masters = zip(*batch)

        xs_long = torch.stack(xs_long)
        xs_short = torch.stack(xs_short)
        true_ks = torch.tensor(true_ks, dtype=torch.long)
        comps = torch.stack(comps)
        params = torch.stack(params)
        noise_stds = torch.tensor(noise_stds, dtype=torch.float32)
        x_masters = torch.stack(x_masters)

        return xs_long, xs_short, true_ks, comps, params, noise_stds, x_masters

    else:
        raise ValueError(f"Unexpected sample length {sample_len}. Expected 6 or 7.")


class MultiSinusoidDataset(Dataset):
    """
    Synthetic dataset of sinusoid mixtures with shared noise.

    Modes
    -----
    - "train": fast, analytic signals on long/short grids, resampled noise only.
    - "debug": same as train, but also returns master signal for plotting/consistency checks.
    - "inference": build full master signal (signal+noise) once and resample it to long/short.
    """

    def __init__(
        self,
        set_size=5000,
        num_samples_long=10000,
        tEnd_long=1000,
        num_samples_short=5120,
        tEnd_short=10,
        max_components=3,
        freq_range=(2.5, 3.0),
        amp_range=(0.5, 1.5),
        min_freq_sep=0.01,
        noise_std=(0.1, 0.3),
        seed=None,
        num_comp=None,
        mode: str = "train",
        allowed_K_values: list[int] | None = None,  # 👈 new argument
    ):
        self.set_size = set_size
        self.num_samples_long = num_samples_long
        self.tEnd_long = tEnd_long
        self.num_samples_short = num_samples_short
        self.tEnd_short = tEnd_short
        self.max_components = max_components
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.min_freq_sep = min_freq_sep
        self.noise_std = noise_std
        self.seed = seed
        self.num_comp = num_comp
        self.mode = mode.lower()
        self.allowed_K_values = allowed_K_values  # 👈 store list
        assert self.mode in ("train", "debug", "inference")

        # === time axes ===
        self.t_long = torch.linspace(0, tEnd_long, num_samples_long)
        self.t_short = torch.linspace(0, tEnd_short, num_samples_short)
        self.dt_high = self.tEnd_short / self.num_samples_short
        self.num_samples_master = int(round(self.tEnd_long / self.dt_high))
        self.t_master = torch.linspace(0, tEnd_long, self.num_samples_master)
        self.dt_master = float(self.t_master[1] - self.t_master[0])
        self.dt_long = float(self.t_long[1] - self.t_long[0])
        self.dt_short = float(self.t_short[1] - self.t_short[0])
        self.dt_shift_long = self.dt_long - self.dt_master
        self.dt_shift_short = self.dt_short - self.dt_master
        self.dt_shift = 0.5 * (self.dt_shift_long + self.dt_shift_short)

    def __len__(self):
        return self.set_size

    def __getitem__(self, idx):
        if self.seed is not None:
            seed_val = self.seed + idx
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            random.seed(seed_val)

        # --- number of components ---
        if self.num_comp is not None:
            K = self.num_comp
        elif self.allowed_K_values is not None and len(self.allowed_K_values) > 0:
            K = int(
                random.choice(self.allowed_K_values)
            )  # 👈 sample from provided list
        else:
            K = torch.randint(1, self.max_components + 1, ()).item()

        # --- sample freqs with min separation ---
        freqs, attempts = [], 0
        while len(freqs) < K and attempts < 100:
            f_try = float(torch.empty(1).uniform_(*self.freq_range))
            if all(abs(f_try - fe) >= self.min_freq_sep for fe in freqs):
                freqs.append(f_try)
            attempts += 1
        if len(freqs) == 0:
            freqs = [float(torch.empty(1).uniform_(*self.freq_range))]
        K = len(freqs)

        # amplitudes, phases
        A = torch.empty(K).uniform_(*self.amp_range)
        phi = torch.empty(K).uniform_(0.0, 2 * math.pi)
        f = torch.tensor(freqs)

        noise_scale = float(torch.empty(1).uniform_(*self.noise_std))

        if self.mode in ("train", "debug"):
            # analytic signals
            comps_long = A[:, None] * torch.sin(
                2 * math.pi * f[:, None] * self.t_long[None, :] + phi[:, None]
            )
            comps_short = A[:, None] * torch.sin(
                2 * math.pi * f[:, None] * self.t_short[None, :] + phi[:, None]
            )
            x_long = comps_long.sum(dim=0)
            x_short = comps_short.sum(dim=0)

            x_master = None
            if noise_scale > 0.0:
                noise_master = torch.randn(self.num_samples_master) * noise_scale
                noise_long = F.interpolate(
                    noise_master[None, None, :],
                    size=self.num_samples_long,
                    mode="linear",
                    align_corners=False,
                ).squeeze()
                noise_short = F.interpolate(
                    noise_master[None, None, :],
                    size=self.num_samples_short,
                    mode="linear",
                    align_corners=False,
                ).squeeze()
                x_long += noise_long
                x_short += noise_short

                if self.mode == "debug":
                    comps_master = A[:, None] * torch.sin(
                        2 * math.pi * f[:, None] * self.t_master[None, :] + phi[:, None]
                    )
                    x_master = comps_master.sum(dim=0) + noise_master

        # -------------------------------------------------------------------------
        # Note on data generation in inference mode:
        # -------------------------------------------------------------------------
        # Our initial goal was to mimic the realistic situation where we have a single
        # long, high-resolution master stream (signal + noise), from which we derive:
        #   (1) a long-term low-resolution stream by downsampling, and
        #   (2) a short-term high-resolution stream by cutting out a contiguous window.
        #
        # When we first implemented this with F.interpolate (linear resampling), we
        # observed systematic phase shifts: the interpolation kernel introduces an
        # effective time offset that becomes frequency-dependent in phase.
        #
        # To fix this, we now generate the *signal* analytically on each grid
        # (master/long/short) to ensure exact phase alignment, while the *noise* is
        # still drawn on the master grid and consistently resampled. This preserves
        # both realism (all noise comes from one master stream) and calibration
        # (no artificial phase bias in the signal).
        #
        # For strict physical fidelity, one could instead decimate the full master
        # (signal + noise) using a high-quality sinc/FIR resampler, but the current
        # approach is sufficient and exact as long as signal frequencies remain well
        # below Nyquist.
        # -------------------------------------------------------------------------
        elif self.mode == "inference":
            # --- analytic signals on each grid (no phase error) ---
            comps_master = A[:, None] * torch.sin(
                2 * math.pi * f[:, None] * self.t_master[None, :] + phi[:, None]
            )
            comps_long = A[:, None] * torch.sin(
                2 * math.pi * f[:, None] * self.t_long[None, :] + phi[:, None]
            )
            comps_short = A[:, None] * torch.sin(
                2 * math.pi * f[:, None] * self.t_short[None, :] + phi[:, None]
            )

            # --- noise only on master, then resample ---
            noise_master = (
                torch.randn(self.num_samples_master) * noise_scale
                if noise_scale > 0
                else torch.zeros_like(self.t_master)
            )
            n_long = F.interpolate(
                noise_master[None, None, :],
                size=self.num_samples_long,
                mode="linear",
                align_corners=False,
            ).squeeze()
            n_short = F.interpolate(
                noise_master[: int(round(self.tEnd_short / self.dt_high))][
                    None, None, :
                ],
                size=self.num_samples_short,
                mode="linear",
                align_corners=False,
            ).squeeze()

            # --- build final signals ---
            x_master = comps_master.sum(dim=0) + noise_master
            x_long = comps_long.sum(dim=0) + n_long
            x_short = comps_short.sum(dim=0) + n_short

        # pad components/params
        if K < self.max_components:
            pad_c = torch.zeros(self.max_components - K, self.num_samples_long)
            comps_long = torch.cat([comps_long, pad_c], dim=0)
            pad_p = torch.zeros(self.max_components - K, 3)
            params = torch.cat([torch.stack([A, phi, f], dim=1), pad_p], dim=0)
        else:
            params = torch.stack([A, phi, f], dim=1)

        params = params.float()

        # return
        if self.mode == "train":
            return (
                x_long.float(),
                x_short.float(),
                K,
                comps_long.float(),
                params,
                float(noise_scale),
            )
        else:  # debug & inference
            return (
                x_long.float(),
                x_short.float(),
                K,
                comps_long.float(),
                params,
                float(noise_scale),
                x_master.float(),
            )
