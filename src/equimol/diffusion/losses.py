from __future__ import annotations

import torch


def coordinate_noise_mse(
    eps_hat: torch.Tensor,
    eps: torch.Tensor,
    node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return MSE between predicted and target coordinate noise.

    Objective:
        Train an epsilon-prediction denoiser by comparing predicted coordinate
        noise with the Gaussian noise used in the forward process.

    Shapes:
        - eps_hat: [N, D]
        - eps: [N, D]
        - node_mask: [N] or [N, 1] optional
        - output: [] scalar loss

    Equation:
        - loss = mean_i ||eps_hat_i - eps_i||^2
        - with mask: loss = sum_i mask_i ||eps_hat_i - eps_i||^2 / sum_i mask_i

    """
    if eps_hat.ndim != 2:
        raise ValueError(
            f"Expected eps_hat with shape [N, D], got {tuple(eps_hat.shape)}"
        )
    if eps.shape != eps_hat.shape:
        raise ValueError(
            f"Expected eps with shape {tuple(eps_hat.shape)}, got {tuple(eps.shape)}"
        )
    if eps.device != eps_hat.device:
        raise ValueError("eps must be on the same device as eps_hat")
    if not torch.is_floating_point(eps_hat):
        raise TypeError(f"eps_hat must be floating point, got {eps_hat.dtype}")
    if not torch.is_floating_point(eps):
        raise TypeError(f"eps must be floating point, got {eps.dtype}")

    per_node_loss = (eps_hat - eps).square().sum(dim=-1)
    if node_mask is None:
        return per_node_loss.mean()

    if node_mask.device != eps_hat.device:
        raise ValueError("node_mask must be on the same device as eps_hat")
    if node_mask.ndim == 2 and node_mask.shape[-1] == 1:
        node_mask = node_mask.squeeze(-1)
    if node_mask.ndim != 1 or node_mask.shape[0] != eps_hat.shape[0]:
        raise ValueError(
            f"Expected node_mask with shape [N] or [N, 1], got {tuple(node_mask.shape)}"
        )

    weights = node_mask.to(dtype=eps_hat.dtype)
    denom = weights.sum()
    if denom <= 0:
        raise ValueError("node_mask must contain at least one valid node")
    return (per_node_loss * weights).sum() / denom
