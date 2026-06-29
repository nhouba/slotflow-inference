"""
Clustering Gaussian-mixture dataset for the GMM benchmark.

  - Dirichlet component weights (small clusters are genuinely hard -> cardinality
    ambiguity),
  - a range of separations including overlapping/hard cases,
  - diagonal covariances,
  - on-the-fly infinite data (seed=None).

Total points per set is FIXED (default collate works); per-component counts vary
via Multinomial(weights) with a minimum per component for identifiability. Variable
total N + masking is an optional extension, not done here.

Returns (X, K, params): X (n_points, dim); K int; params (k_max, 4) padded,
rows [mu_x, mu_y, log s_x, log s_y].
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset


class GMMDataset(Dataset):
    def __init__(self, set_size=1_000_000, k_max=10, dim=2, n_points=400,
                 min_points=5, mean_range=(-6.0, 6.0), logscale_range=(-1.2, 0.0),
                 sep_range=(0.5, 3.0), dirichlet_alpha=1.0, seed=None):
        self.set_size = set_size
        self.k_max = k_max
        self.dim = dim
        self.n_points = n_points
        self.min_points = min_points
        self.mean_range = mean_range
        self.logscale_range = logscale_range
        self.sep_range = sep_range
        self.dirichlet_alpha = dirichlet_alpha
        self.seed = seed

    def __len__(self):
        return self.set_size

    def __getitem__(self, idx):
        if self.seed is not None:
            s = self.seed + idx
            np.random.seed(s)
            torch.manual_seed(s)
            random.seed(s)

        K = int(torch.randint(1, self.k_max + 1, ()).item())

        # weights (clamped so logit is finite, incl. the degenerate K=1 case)
        w = torch.distributions.Dirichlet(self.dirichlet_alpha * torch.ones(K)).sample()
        w = w.clamp(1e-3, 1 - 1e-3)

        # per-component diagonal sigmas
        log_s = torch.empty(K, self.dim).uniform_(*self.logscale_range)
        sigmas = torch.exp(log_s)
        mean_sigma = float(sigmas.mean().item())
        sep = float(torch.empty(1).uniform_(*self.sep_range).item()) * mean_sigma

        # well-/poorly-separated means via rejection
        means, tries = [], 0
        while len(means) < K and tries < 500:
            m = torch.empty(self.dim).uniform_(*self.mean_range)
            if all(torch.norm(m - me) >= sep for me in means):
                means.append(m)
            tries += 1
        while len(means) < K:  # tight-sep fallback
            means.append(torch.empty(self.dim).uniform_(*self.mean_range))
        means = torch.stack(means)

        # per-component counts: min_points each, remainder ~ Multinomial(w)
        rest = self.n_points - K * self.min_points
        if rest >= 0:
            extra = torch.multinomial(w, rest, replacement=True)
            counts = torch.full((K,), self.min_points, dtype=torch.long)
            counts += torch.bincount(extra, minlength=K)
        else:  # n_points too small for K -> spread as evenly as possible
            counts = torch.full((K,), self.n_points // K, dtype=torch.long)
            counts[: self.n_points - int(counts.sum().item())] += 1

        pts = [means[k] + sigmas[k] * torch.randn(int(counts[k].item()), self.dim)
               for k in range(K)]
        X = torch.cat(pts, dim=0)
        X = X[torch.randperm(X.size(0))]  # order carries no information

        # param_dim=4: theta = [mu_x, mu_y, log s_x, log s_y]. Weights are generative
        # only (variable cluster sizes -> cardinality ambiguity), NOT a flow target
        # (their logit has heavy tails and the simplex constraint is mean-field-broken).
        theta = torch.cat([means, log_s], dim=-1)  # (K, 4)
        if K < self.k_max:
            theta = torch.cat([theta, torch.zeros(self.k_max - K, theta.size(-1))], dim=0)

        return X.float(), K, theta.float()
