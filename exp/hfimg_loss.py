"""High-frequency image supervision from FReSCO Eq. (4)-(7).

The paper specifies ``sigma1 < sigma2`` and the symbolic ``lambda_h1/h2`` but
does not publish their numeric values. Configured values remain experimental.
"""

import torch
import torch.nn.functional as F

from exp.lfimg_loss import _to_bchw, gaussian_lowpass


def difference_of_gaussians(image, kernel_size_1, sigma_1, kernel_size_2, sigma_2):
    if sigma_1 >= sigma_2:
        raise ValueError(f"DoG requires sigma_1 < sigma_2, got {sigma_1} >= {sigma_2}")
    return gaussian_lowpass(image, kernel_size_1, sigma_1) - gaussian_lowpass(
        image, kernel_size_2, sigma_2
    )


def normalized_gt_gradient(gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-image normalized RGB gradient magnitude used by Eq. (5)."""
    gt = _to_bchw(gt)
    dx = F.pad(gt[..., 1:] - gt[..., :-1], (0, 1, 0, 0))
    dy = F.pad(gt[..., 1:, :] - gt[..., :-1, :], (0, 0, 0, 1))
    magnitude = torch.sqrt((dx.square() + dy.square()).clamp_min(0.0)).mean(dim=1, keepdim=True)
    scale = magnitude.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(eps)
    return (magnitude / scale).clamp(0.0, 1.0)


def gradient_difference_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Eq. (6): L1 distance between 2D finite-difference magnitudes."""
    pred = _to_bchw(pred)
    gt = _to_bchw(gt)
    pred_dx = F.pad(pred[..., 1:] - pred[..., :-1], (0, 1, 0, 0))
    gt_dx = F.pad(gt[..., 1:] - gt[..., :-1], (0, 1, 0, 0))
    pred_dy = F.pad(pred[..., 1:, :] - pred[..., :-1, :], (0, 0, 0, 1))
    gt_dy = F.pad(gt[..., 1:, :] - gt[..., :-1, :], (0, 0, 0, 1))
    # Equal epsilon terms cancel in flat regions while avoiding sqrt'(0)=Inf
    # and resulting NaN gradients through constant-color areas.
    eps = 1e-12
    pred_magnitude = torch.sqrt(pred_dx.square() + pred_dy.square() + eps)
    gt_magnitude = torch.sqrt(gt_dx.square() + gt_dy.square() + eps)
    return (pred_magnitude - gt_magnitude).abs().mean()


def compute_hfimg_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    mask: torch.Tensor = None,
    cfg: dict = None,
    step: int = 0,
    report_dir: str = None,
    **unused,
):
    """Compute edge-weighted DoG alignment plus GDL.

    ``mask`` is intentionally not used: Eq. (5) defines its weighting from the
    normalized GT gradient rather than a thresholded or dilated edge band.
    """
    del mask, step, report_dir, unused
    cfg = cfg or {}
    pred = _to_bchw(pred_rgb).float()
    gt = _to_bchw(gt_rgb).to(device=pred.device, dtype=pred.dtype)
    if gt.numel() and gt.detach().max() > 1.0:
        gt = gt / 255.0
    if pred.shape != gt.shape:
        raise ValueError(f"HF image shape mismatch: pred={tuple(pred.shape)}, gt={tuple(gt.shape)}")

    required = ("sigma1", "sigma2", "kernel1", "kernel2", "lambda_h1", "lambda_h2")
    missing = [key for key in required if key not in cfg or cfg[key] is None]
    if missing:
        raise ValueError(f"HF-IMG missing required parameters: {', '.join(missing)}")
    sigma_1 = float(cfg["sigma1"])
    sigma_2 = float(cfg["sigma2"])
    kernel_1 = int(cfg["kernel1"])
    kernel_2 = int(cfg["kernel2"])
    lambda_h1 = float(cfg["lambda_h1"])
    lambda_h2 = float(cfg["lambda_h2"])
    pred_hf = difference_of_gaussians(pred, kernel_1, sigma_1, kernel_2, sigma_2)
    gt_hf = difference_of_gaussians(gt, kernel_1, sigma_1, kernel_2, sigma_2)
    edge_mask = normalized_gt_gradient(gt).detach()
    edge_hf = (edge_mask * (pred_hf - gt_hf).abs()).mean()
    gdl = gradient_difference_loss(pred, gt)
    loss = lambda_h1 * edge_hf + lambda_h2 * gdl
    if not torch.isfinite(loss):
        raise RuntimeError("HF image loss produced NaN or Inf")
    return {
        "loss": loss, "raw": loss.detach(), "edge_hf": edge_hf.detach(), "gdl": gdl.detach(),
        "edge_coverage": float(edge_mask.mean().detach()), "face_coverage": 1.0,
        "sigma1": sigma_1, "sigma2": sigma_2, "kernel1": kernel_1, "kernel2": kernel_2,
        "lambda_h1": lambda_h1, "lambda_h2": lambda_h2, "mode": "dog_edge_gdl",
        "parameter_status": "sigma1/sigma2/lambda_h1/lambda_h2 require experimental validation",
    }
