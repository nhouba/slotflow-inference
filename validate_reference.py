"""
Known-answer validation of the Gibbs reference (baselines/reference_gmm.py).

Generates a synthetic, well-separated K-component GMM with KNOWN means, runs the
Gibbs sampler, relabels to the truth (as Eval-GMM does), and checks the posterior
means recover the truth. If this FAILS, the W2-vs-reference numbers in the
benchmark are not trustworthy and the sampler needs work before they're cited.

Run (no GPU needed): python validate_reference.py
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from baselines.reference_gmm import reference_samples


def relabel_to_truth(samples, true_means):
    out = samples.copy()
    K = samples.shape[1]
    for t in range(samples.shape[0]):
        r, c = linear_sum_assignment(cdist(samples[t, :, :2], true_means))
        order = np.empty(K, dtype=int)
        order[c] = r
        out[t] = samples[t][order]
    return out


def main():
    rng = np.random.default_rng(0)
    K = 3
    true_means = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    true_sig = np.array([0.4, 0.5, 0.45])
    w = np.array([0.4, 0.35, 0.25])
    N = 600
    z = rng.choice(K, N, p=w)
    X = true_means[z] + true_sig[z, None] * rng.normal(size=(N, 2))

    ref = reference_samples(X, K, seed=0)            # (S, K, 4)
    rel = relabel_to_truth(ref, true_means)
    post_mean_mu = rel.mean(0)[:, :2]                # (K, 2)
    post_mean_logsig = rel.mean(0)[:, 2:4]

    mu_err = np.abs(post_mean_mu - true_means).max()
    sig_err = np.abs(np.exp(post_mean_logsig) - true_sig[:, None]).max()

    print("=== Gibbs reference validation (well-separated K=3) ===")
    print("recovered means:\n", post_mean_mu.round(2))
    print("true means:\n", true_means)
    print(f"max |posterior-mean mu - true mu|     = {mu_err:.3f}   (sep ~ 4)")
    print(f"max |posterior-mean sigma - true sigma| = {sig_err:.3f}")
    ok = mu_err < 0.5 and sig_err < 0.3
    print("PASS -- reference is reliable, W2 numbers can be trusted" if ok
          else "FAIL -- Gibbs reference unreliable; DO NOT cite W2 until fixed")


if __name__ == "__main__":
    main()
