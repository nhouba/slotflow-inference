"""
Reference posterior for the GMM benchmark (correctness baseline).

Self-contained (NumPy) blocked Gibbs sampler for a diagonal-covariance Gaussian
mixture with conjugate priors, giving the reference posterior over component
parameters (mu, sigma, w | x, K) for the Wasserstein comparison, plus a BIC-based
model-order selection for an approximate p(K | x).

Scope:
  - The per-K Gibbs sampler is the reference for (mu, sigma, w | x, K). Label
    switching is irrelevant because the evaluation Hungarian-matches components to
    the truth.
  - p(K | x) via BIC is an approximation, used as a classical model-order reference
    rather than a gold standard. validate_reference.py checks the sampler on a
    known-answer synthetic case.

Output param order matches SlotFlow: [mu_x, mu_y, log s_x, log s_y, logit w].
"""

import numpy as np


def gibbs_gmm(X, K, n_iter=1500, burn=750, thin=3, alpha=1.0,
              kappa0=0.01, a0=2.0, b0=0.5, seed=0):
    """Blocked Gibbs for a K-component diagonal-cov GMM. Returns dict of sample arrays:
    mu (S,K,D), logsig (S,K,D), logitw (S,K)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    mu0 = X.mean(0)

    mu = X[rng.choice(N, K, replace=False)].copy()      # (K, D)
    sig2 = np.full((K, D), float(X.var(0).mean()) + 1e-6)
    w = np.full(K, 1.0 / K)

    out = {"mu": [], "logsig": [], "logitw": []}
    for it in range(n_iter):
        # responsibilities  log p(x_i|k) = sum_d N(x_id; mu_kd, sig2_kd)
        diff2 = (X[:, None, :] - mu[None, :, :]) ** 2               # (N,K,D)
        logN = -0.5 * (diff2 / sig2[None] + np.log(2 * np.pi * sig2[None])).sum(-1)  # (N,K)
        logr = np.log(w)[None, :] + logN
        logr -= logr.max(1, keepdims=True)
        r = np.exp(logr)
        r /= r.sum(1, keepdims=True)
        # sample assignments
        u = rng.random(N)
        z = (r.cumsum(1) < u[:, None]).sum(1)
        z = np.clip(z, 0, K - 1)

        nk = np.bincount(z, minlength=K)
        w = rng.dirichlet(alpha + nk)

        for k in range(K):
            Xk = X[z == k]
            nkk = Xk.shape[0]
            if nkk == 0:
                mu[k] = X[rng.integers(N)]
                sig2[k] = 1.0 / rng.gamma(a0, 1.0 / b0, size=D)
                continue
            prec = kappa0 + nkk / sig2[k]                          # (D,)
            mean = (kappa0 * mu0 + Xk.sum(0) / sig2[k]) / prec
            mu[k] = mean + rng.normal(size=D) / np.sqrt(prec)
            a = a0 + nkk / 2.0
            b = b0 + 0.5 * ((Xk - mu[k]) ** 2).sum(0)              # (D,)
            sig2[k] = 1.0 / rng.gamma(a, 1.0 / b, size=D)

        if it >= burn and (it - burn) % thin == 0:
            out["mu"].append(mu.copy())
            out["logsig"].append(0.5 * np.log(sig2).copy())
            w_c = np.clip(w, 1e-6, 1 - 1e-6)
            out["logitw"].append((np.log(w_c) - np.log1p(-w_c)).copy())
    return {k: np.asarray(v) for k, v in out.items()}


def reference_samples(X, K, **kw):
    """(S, K, 4) reference samples in SlotFlow param order [mu_x, mu_y, log sx, log sy].

    Weights are sampled internally (needed for the mixture) but NOT returned --
    param_dim=4 drops the weight from the comparison. Components are label-switched
    across Gibbs iterations; the caller (Eval-GMM) relabels to the truth.
    """
    g = gibbs_gmm(X, K, **kw)
    return np.concatenate([g["mu"], g["logsig"]], axis=-1)


def _mixture_loglik(X, mu, sig2, w):
    diff2 = (X[:, None, :] - mu[None]) ** 2
    logN = -0.5 * (diff2 / sig2[None] + np.log(2 * np.pi * sig2[None])).sum(-1)  # (N,K)
    ll = np.log(w)[None] + logN
    m = ll.max(1, keepdims=True)
    return (m.squeeze(1) + np.log(np.exp(ll - m).sum(1))).sum()


def bic_model_selection(X, k_range, **kw):
    """Approximate p(K|x) via BIC at the posterior mean. Returns (pk_dict, bic_dict)."""
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    bics = {}
    for K in k_range:
        g = gibbs_gmm(X, K, **kw)
        mu = g["mu"].mean(0)
        sig2 = np.exp(2 * g["logsig"].mean(0))
        w = 1.0 / (1.0 + np.exp(-g["logitw"].mean(0)))
        w = w / w.sum()
        loglik = _mixture_loglik(X, mu, sig2, w)
        n_params = K * (2 * D) + (K - 1)
        bics[K] = -2.0 * loglik + n_params * np.log(N)
    ks = np.array(list(bics))
    vals = np.array([bics[k] for k in ks])
    pk = np.exp(-0.5 * (vals - vals.min()))
    pk /= pk.sum()
    return dict(zip(ks.tolist(), pk.tolist())), bics
