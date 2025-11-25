import torch
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
import torch


def hungarian_flow_matching_loss(
    flow,
    context,
    true_params,
    true_ks,
    device="cpu",
    phase_weight=2.0,
    freq_weight=3.0,
    freq_range=(2.5, 3.0),
):
    """
    Flow-based Hungarian matching loss with extra weight on phase (cosφ, sinφ).

    Args:
        flow: Normalizing flow model
        context: torch.Tensor, concatenated context slots
        true_params: list of true parameter tensors per sample, shape (k_b, 4)
                     where each param = [amp, cosφ, sinφ, freq]
        true_ks: tensor of number of components per batch element
        device: device for computation
        phase_weight: multiplicative factor for (cosφ, sinφ) dimensions
    """
    B = len(true_ks)
    all_ctx_flat, all_gt_flat, batch_info = [], [], []
    slot_ptr = 0

    # Pre-compute all flow evaluations in one batch
    for b in range(B):
        k_b = true_ks[b].item()
        gt = true_params[b]  # (k_b, 4)
        gt[:, 3] = gt[:, 3] - (freq_range[0] + freq_range[1]) / 2  # residual frequency

        # --- Apply phase weighting ---
        gt_weighted = gt.clone()
        gt_weighted[:, 1:3] *= phase_weight  # scale cosφ and sinφ
        gt_weighted[:, 3] *= freq_weight  # scale freq

        ctx = context[slot_ptr : slot_ptr + k_b]  # (k_b, context_dim)

        # Create all pairwise combinations
        ctx_exp = ctx.unsqueeze(0).expand(k_b, -1, -1)
        gt_exp = gt_weighted.unsqueeze(1).expand(-1, k_b, -1)
        ctx_flat = ctx_exp.reshape(-1, ctx.shape[-1])
        gt_flat = gt_exp.reshape(-1, gt_weighted.shape[-1])

        all_ctx_flat.append(ctx_flat)
        all_gt_flat.append(gt_flat)
        batch_info.append((b, k_b))
        slot_ptr += k_b

    # Single batched flow evaluation
    all_ctx = torch.cat(all_ctx_flat, dim=0)
    all_gt = torch.cat(all_gt_flat, dim=0)
    all_logps = flow.log_prob(all_gt, context=all_ctx)

    # Process results
    weighted_sum, total_slots, logp_ptr = 0.0, 0, 0
    for b, k_b in batch_info:
        num_pairs = k_b * k_b
        cost = -all_logps[logp_ptr : logp_ptr + num_pairs].view(k_b, k_b)
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        matched_logps = cost[row, col].mean()  # per-sample mean
        weighted_sum += matched_logps * k_b  # weight by k_b
        total_slots += k_b
        logp_ptr += num_pairs

    return weighted_sum / total_slots


def noise_supervision_loss(pred_std, true_std):
    """
    Computes the mean squared error (MSE) loss between the predicted and true standard deviation
    of Gaussian noise, measured in the log-space.

    This loss function is used to supervise the model's prediction of the log-standard deviation
    of Gaussian noise, comparing it to the ground-truth values.

    Args:
        pred_std (Tensor)
        true_std (Tensor)

    Returns:
        torch.Tensor: Scalar loss value representing the mean squared error between
                      the predicted and true log-standard deviations.
    """
    true_std = true_std.to(pred_std.device)
    return F.mse_loss(pred_std, true_std)
