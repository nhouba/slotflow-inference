"""
Permutation-invariant set encoders for the GMM instantiation.

Default: a Set Transformer (Lee et al., 2019) -- ISAB blocks (linear in N via
induced points) + PMA pooling. Fallback: Deep Sets.

Both return the two embeddings the shared core consumes, at dim 2*hidden_dim:
    g     -> conditions the shared flow + slot contexts
    e_cls -> feeds the K-classifier
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Set Transformer building blocks ----------------
class MAB(nn.Module):
    """Multihead attention block."""

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln0 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K2, V = self.fc_k(K), self.fc_v(K)
        d = self.dim_V // self.num_heads
        Qh = torch.cat(Q.split(d, 2), 0)
        Kh = torch.cat(K2.split(d, 2), 0)
        Vh = torch.cat(V.split(d, 2), 0)
        A = torch.softmax(Qh.bmm(Kh.transpose(1, 2)) / math.sqrt(d), 2)
        O = torch.cat((Qh + A.bmm(Vh)).split(Q.size(0), 0), 2)
        O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O)
        return O


class ISAB(nn.Module):
    """Induced set attention block (linear in set size)."""

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    """Pooling by multihead attention (k seed vectors)."""

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformerEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, num_heads=4, num_inds=16, ln=True):
        super().__init__()
        d = 2 * hidden_dim
        self.enc = nn.Sequential(
            ISAB(in_dim, d, num_heads, num_inds, ln),
            ISAB(d, d, num_heads, num_inds, ln),
        )
        self.pma_g = PMA(d, num_heads, 1, ln)
        self.pma_cls = PMA(d, num_heads, 1, ln)

    def forward(self, X):  # X: (B, N, in_dim)
        H = self.enc(X)
        g = self.pma_g(H).squeeze(1)        # (B, 2*hidden_dim)
        e_cls = self.pma_cls(H).squeeze(1)  # (B, 2*hidden_dim)
        return g, e_cls


# ---------------- Deep Sets fallback ----------------
class DeepSetsEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, phi_width=256, rho_width=256):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, phi_width), nn.GELU(),
            nn.Linear(phi_width, phi_width), nn.GELU(),
            nn.Linear(phi_width, phi_width), nn.GELU())
        self.rho_flow = nn.Sequential(
            nn.Linear(2 * phi_width, rho_width), nn.GELU(),
            nn.Linear(rho_width, 2 * hidden_dim))
        self.rho_cls = nn.Sequential(
            nn.Linear(2 * phi_width, rho_width), nn.GELU(),
            nn.Linear(rho_width, 2 * hidden_dim))

    def forward(self, X):
        h = self.phi(X)
        pooled = torch.cat([h.mean(1), h.max(1).values], dim=-1)
        return self.rho_flow(pooled), self.rho_cls(pooled)
