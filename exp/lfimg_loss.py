"""Low-frequency image supervision from FReSCO Eq. (2)-(3)."""

import torch
import torch.nn.functional as F


def _to_bchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected a 4D image tensor, got {tuple(image.shape)}")
    if image.shape[1] == 3:
        return image
    if image.shape[-1] == 3:
        return image.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Expected RGB image in BCHW or BHWC layout, got {tuple(image.shape)}")


def gaussian_kernel_2d(kernel_size: int, sigma: float, reference: torch.Tensor) -> torch.Tensor:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    coords = torch.arange(kernel_size, device=reference.device, dtype=reference.dtype)
    coords = coords - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def gaussian_lowpass(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    image = _to_bchw(image)
    kernel = gaussian_kernel_2d(kernel_size, sigma, image).expand(image.shape[1], 1, -1, -1)
    pad = kernel_size // 2
    padded = F.pad(image, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, kernel, groups=image.shape[1])


def compute_lfimg_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    mask: torch.Tensor = None,
    cfg: dict = None,
    step: int = 0,
    report_dir: str = None,
):
    """Compute ``mean(|G_sigma(pred) - G_sigma(gt)|)``.

    The paper defines the loss on the complete rendered head image. ``mask`` is
    accepted for unit-level reuse; the trainer intentionally passes None.
    """
    del step, report_dir
    cfg = cfg or {}
    pred = _to_bchw(pred_rgb).float()
    gt = _to_bchw(gt_rgb).to(device=pred.device, dtype=pred.dtype)
    if gt.numel() and gt.detach().max() > 1.0:
        gt = gt / 255.0
    if pred.shape != gt.shape:
        raise ValueError(f"LF image shape mismatch: pred={tuple(pred.shape)}, gt={tuple(gt.shape)}")

    kernel_size = int(cfg.get("kernel", 9))
    sigma = float(cfg.get("sigma", kernel_size / 6.0))
    if sigma <= 0:
        sigma = kernel_size / 6.0
    pred_lf = gaussian_lowpass(pred, kernel_size, sigma)
    gt_lf = gaussian_lowpass(gt, kernel_size, sigma)
    diff = (pred_lf - gt_lf).abs()

    if mask is None:
        loss = diff.mean()
        coverage = 1.0
        denom = float(diff.numel())
    else:
        if mask.ndim == 3:
            mask = mask[:, None]
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask.permute(0, 3, 1, 2)
        if mask.ndim != 4 or mask.shape[1] != 1 or mask.shape[0] != pred.shape[0] or mask.shape[-2:] != pred.shape[-2:]:
            raise ValueError(f"Invalid LF mask shape {tuple(mask.shape)} for image {tuple(pred.shape)}")
        weight = mask.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
        denominator = (weight.sum() * pred.shape[1]).clamp_min(1.0)
        loss = (diff * weight).sum() / denominator
        coverage = float(weight.mean().detach())
        denom = float(denominator.detach())

    if not torch.isfinite(loss):
        raise RuntimeError("LF image loss produced NaN or Inf")
    return {"loss": loss, "raw": loss.detach(), "cov": coverage, "denom": denom,
            "mode": "l1", "k": kernel_size, "sigma": sigma}
