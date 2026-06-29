"""
SlotFlowGMM -- the GMM instantiation, built on the SHARED core (src/core.py).

Only the encoder and the parameterization (param_dim=4: mu_x, mu_y, log s_x,
log s_y) are GMM-specific; the classifier, slot mechanism, and shared flow are the
unchanged framework core.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import build_conditional_flow, build_slot_contexts
from src.gmm.encoder import DeepSetsEncoder, SetTransformerEncoder


class SlotFlowGMM(nn.Module):
    def __init__(self, hidden_dim=128, max_slots=10, flow_depth=8, param_dim=4,
                 in_dim=2, encoder="set_transformer"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_slots = max_slots
        if encoder == "set_transformer":
            self.encoder = SetTransformerEncoder(in_dim=in_dim, hidden_dim=hidden_dim)
        else:
            self.encoder = DeepSetsEncoder(in_dim=in_dim, hidden_dim=hidden_dim)
        self.k_classifier = nn.Linear(2 * hidden_dim, max_slots)          # SAME as core
        self.flow_context_dim = 2 * hidden_dim + max_slots                 # SAME formula
        self.flow = build_conditional_flow(param_dim, self.flow_context_dim, flow_depth)

    def forward(self, X, use_gt_k=None):
        g, e_cls = self.encoder(X)                                         # NEW (encoder)
        k_logits = self.k_classifier(e_cls)                                # SAME
        k_pred = torch.argmax(F.softmax(k_logits, dim=-1), dim=-1) + 1
        K = use_gt_k if use_gt_k is not None else k_pred
        context, batch_slot_ids = build_slot_contexts(g, K, self.max_slots, X.device)
        return {
            "K_logits": k_logits,
            "K_pred": K,
            "context": context,
            "batch_slot_ids": batch_slot_ids,
            "h_embed_flow": g,
        }
