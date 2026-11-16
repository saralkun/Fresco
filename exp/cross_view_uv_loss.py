"""Cross-view rendered-to-rendered UV consistency, FReSCO Eq. (10)-(11)."""

import torch


def bake_attribute_to_uv(attribute, face_id, face_bw, vertex_uvs, faces_uvs, px_mask, uv_hw):
    """Nearest-texel UV splat with differentiable accumulation of attributes."""
    if attribute.ndim != 4:
        raise ValueError(f"attribute must be BHWC, got {tuple(attribute.shape)}")
    batch, _, _, channels = attribute.shape
    uv_h, uv_w = map(int, uv_hw)
    baked = []
    masks = []
    for b in range(batch):
        valid = px_mask[b].bool() & (face_id[b] >= 0)
        ys, xs = torch.where(valid)
        flat = attribute.new_zeros(channels, uv_h * uv_w)
        count = attribute.new_zeros(1, uv_h * uv_w)
        if ys.numel():
            triangle = face_id[b, ys, xs].long()
            uv_indices = faces_uvs[triangle]
            tri_uv = vertex_uvs[uv_indices]
            bary = face_bw[b, ys, xs]
            if bary.shape[-1] == 2:
                bary = torch.cat((bary, 1.0 - bary.sum(dim=-1, keepdim=True)), dim=-1)
            elif bary.shape[-1] != 3:
                raise ValueError(f"face_bw must have 2 or 3 channels, got {bary.shape[-1]}")
            uv = (tri_uv * bary[..., None]).sum(dim=1)
            # Match the existing UV baker and seam-map rasterizer: normalized
            # coordinates map to texels by floor/truncation, not rounding.
            uv_x = (uv[:, 0] * (uv_w - 1)).long().clamp(0, uv_w - 1)
            uv_y = (uv[:, 1] * (uv_h - 1)).long().clamp(0, uv_h - 1)
            linear = uv_y * uv_w + uv_x
            values = attribute[b, ys, xs].transpose(0, 1)
            flat.scatter_add_(1, linear[None].expand(channels, -1), values)
            count.scatter_add_(1, linear[None], torch.ones_like(linear, dtype=attribute.dtype)[None])
        count_image = count.view(1, uv_h, uv_w)
        baked.append((flat.view(channels, uv_h, uv_w) / count_image.clamp_min(1.0)))
        masks.append((count_image > 0).to(attribute.dtype))
    return torch.stack(baked), torch.stack(masks)


def build_same_frame_pairs(frame_indices, camera_ids):
    frame_indices = [int(x) for x in torch.as_tensor(frame_indices).cpu().tolist()]
    camera_ids = [str(x) for x in camera_ids]
    pairs = []
    used = set()
    for i, (frame, camera) in enumerate(zip(frame_indices, camera_ids)):
        if i in used:
            continue
        for j in range(i + 1, len(frame_indices)):
            if j not in used and frame_indices[j] == frame and camera_ids[j] != camera:
                pairs.append((i, j))
                used.update((i, j))
                break
    return pairs


def compute_cross_view_uv_loss(uv_render, uv_visibility, uv_confidence, frame_indices, camera_ids):
    """Apply Eq. (10)-(11), detaching only the target view texture."""
    pairs = build_same_frame_pairs(frame_indices, camera_ids)
    zero = uv_render.sum() * 0.0
    if not pairs:
        return {"loss": zero, "coverage": 0.0, "pair_count": 0, "skipped": True,
                "reason": "no_same_timestep_cross_camera_pair", "mean_confidence": 0.0,
                "mutual_texels": 0}
    losses = []
    valid_texels = 0.0
    total_texels = 0.0
    effective_weight_sum = 0.0
    mutual_texel_count = 0.0
    for a, b in pairs:
        mutual = uv_visibility[a] * uv_visibility[b]
        weight = mutual * 0.5 * (uv_confidence[a] + uv_confidence[b])
        denominator = mutual.sum()
        total_texels += float(mutual.numel())
        valid_texels += float((mutual > 0).sum().detach())
        if denominator.detach().item() <= 0:
            continue
        effective_weight_sum += float(weight.sum().detach())
        mutual_texel_count += float(denominator.detach())
        # Eq. (10) divides by the mutually visible set, not by confidence mass.
        losses.append((weight * (uv_render[a] - uv_render[b].detach()).abs()).sum() /
                      (denominator * uv_render.shape[1]).clamp_min(1.0))
    if not losses:
        return {"loss": zero, "coverage": 0.0, "pair_count": len(pairs), "skipped": True,
                "reason": "no_mutual_visibility", "mean_confidence": 0.0,
                "mutual_texels": 0}
    if effective_weight_sum <= 0:
        return {"loss": zero, "coverage": valid_texels / max(total_texels, 1.0),
                "pair_count": len(losses), "skipped": True,
                "reason": "zero_view_confidence", "mean_confidence": 0.0,
                "mutual_texels": int(mutual_texel_count)}
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Cross-view UV loss produced NaN or Inf")
    return {"loss": loss, "coverage": valid_texels / max(total_texels, 1.0),
            "pair_count": len(losses), "skipped": False, "reason": "",
            "mean_confidence": effective_weight_sum / max(mutual_texel_count, 1.0),
            "mutual_texels": int(mutual_texel_count)}
