"""UV seam pair consistency and seam-band TV, FReSCO Eq. (12)."""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def seam_tv_loss(texture: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    dx = (texture[..., 1:] - texture[..., :-1]).abs()
    dy = (texture[..., 1:, :] - texture[..., :-1, :]).abs()
    mask_x = band[..., 1:] * band[..., :-1]
    mask_y = band[..., 1:, :] * band[..., :-1, :]
    denom = ((mask_x.sum() + mask_y.sum()) * texture.shape[1]).clamp_min(1.0)
    return ((dx * mask_x).sum() + (dy * mask_y).sum()) / denom


def compute_seam_loss(
    uv_texture: torch.Tensor,
    config: Dict[str, Any],
    step: int = 0,
    is_main: bool = True,
    uv_mask: Optional[torch.Tensor] = None,
    pair_u: Optional[torch.Tensor] = None,
    pair_v: Optional[torch.Tensor] = None,
    pair_valid: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    del step, is_main
    if uv_texture.ndim != 4 or uv_texture.shape[1] != 3:
        raise ValueError(f"uv_texture must be B3HW, got {tuple(uv_texture.shape)}")
    device, dtype = uv_texture.device, uv_texture.dtype
    zero = uv_texture.sum() * 0.0
    if pair_u is None or pair_v is None or pair_valid is None:
        return {"loss": zero, "pair": zero.detach(), "tv": zero.detach(), "band_cov": 0.0,
                "effective_cov": 0.0, "pair_cov": 0.0, "skipped": True, "reason": "missing_pair_maps"}

    batch, _, height, width = uv_texture.shape
    pair_u = pair_u.to(device=device, dtype=dtype)
    pair_v = pair_v.to(device=device, dtype=dtype)
    pair_valid = pair_valid.to(device=device, dtype=dtype)
    if pair_valid.shape[-2:] != (height, width):
        raise ValueError(f"Seam map resolution {pair_valid.shape[-2:]} != texture {(height, width)}")
    band_px = int(config.get("band_px", 2))
    kernel = 2 * band_px + 1
    band = (F.max_pool2d(pair_valid, kernel, stride=1, padding=band_px) > 0).to(dtype)
    band = band.expand(batch, -1, -1, -1)
    visibility = torch.ones_like(band) if uv_mask is None else uv_mask.to(device=device, dtype=dtype).clamp(0, 1)
    effective = band * visibility.detach()
    band_cov = float(band.mean().detach())
    effective_cov = float(effective.mean().detach())
    if pair_valid.sum().detach().item() == 0:
        return {"loss": zero, "pair": zero.detach(), "tv": zero.detach(), "band_cov": band_cov,
                "effective_cov": effective_cov, "pair_cov": 0.0, "skipped": True, "reason": "empty_seam_map"}

    grid = torch.stack((pair_u, pair_v), dim=-1).expand(batch, -1, -1, -1, -1).squeeze(1)
    partner = F.grid_sample(uv_texture, grid, mode="nearest", padding_mode="zeros", align_corners=True)
    partner_visibility = F.grid_sample(
        visibility.detach(), grid, mode="nearest", padding_mode="zeros", align_corners=True
    )
    pair_weight = pair_valid.expand(batch, -1, -1, -1) * visibility.detach() * partner_visibility
    pair_coverage = float((pair_weight > 0).float().mean().detach())
    pair_denom = (pair_weight.sum() * uv_texture.shape[1]).clamp_min(1.0)
    pair_loss = ((uv_texture - partner).abs() * pair_weight).sum() / pair_denom
    tv_loss = seam_tv_loss(uv_texture, effective)
    weight_pair = float(config.get("weight_pair", config.get("weight", 0.02)))
    weight_tv = float(config.get("weight_tv", 0.004))
    loss = weight_pair * pair_loss + weight_tv * tv_loss
    if not torch.isfinite(loss):
        raise RuntimeError("UV seam loss produced NaN or Inf")
    return {"loss": loss, "pair": pair_loss.detach(), "tv": tv_loss.detach(),
            "band_cov": band_cov, "effective_cov": effective_cov, "pair_cov": pair_coverage,
            "skipped": False, "reason": "", "weight_pair": weight_pair, "weight_tv": weight_tv}
