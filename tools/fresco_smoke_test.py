"""Real-batch FReSCO forward/backward smoke test. Never steps or saves a model."""

import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import yaml
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from dataset.nersemble import NeRSembleData
from exp.config_utils import normalize_dotted_config, validate_hf_image_config
from exp.cross_view_sampler import SameFramePairBatchSampler
from exp.cross_view_uv_loss import build_same_frame_pairs, compute_cross_view_uv_loss
from exp.hfimg_loss import compute_hfimg_loss
from exp.lfimg_loss import compute_lfimg_loss
from exp.uv_seam_loss import compute_seam_loss
from geom.seam_pairs import _build_edge_face_adjacency, _detect_seam_edges
from joint_trainer import JointTrainer


REPORT_HANDLE = None


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ManifestBatchSampler(Sampler):
    def __init__(self, batches):
        self.batches = [list(map(int, batch)) for batch in batches]

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


def frame_identity(dataset, index):
    frame = dataset.framelist[int(index)]
    return {
        "dataset_index": int(index),
        "timestep": int(frame["timestep_index"]),
        "camera_id": str(frame["camera_id"]),
        "camera_index": int(frame["camera_index"]),
        "file_path": str(frame["file_path"]),
    }


def make_batch_manifest(dataset, batches, sampler_mode, batch_size, seed):
    payload = {
        "version": 1,
        "dataset_size": len(dataset),
        "dataset_root": str(dataset.basedir),
        "sampler_mode": sampler_mode,
        "batch_size": int(batch_size),
        "seed": int(seed),
        "batches": [],
    }
    for batch_index, indices in enumerate(batches):
        samples = [frame_identity(dataset, index) for index in indices]
        payload["batches"].append({
            "batch_index": batch_index,
            "dataset_indices": [sample["dataset_index"] for sample in samples],
            "samples": samples,
        })
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def validate_batch_manifest(manifest, dataset, batch_size):
    if manifest.get("version") != 1:
        raise ValueError(f"Unsupported batch manifest version: {manifest.get('version')}")
    if int(manifest.get("dataset_size", -1)) != len(dataset):
        raise ValueError(
            f"Manifest dataset_size={manifest.get('dataset_size')} does not match {len(dataset)}"
        )
    if int(manifest.get("batch_size", -1)) != int(batch_size):
        raise ValueError(
            f"Manifest batch_size={manifest.get('batch_size')} does not match --batch-size={batch_size}"
        )
    unsigned = dict(manifest)
    expected_sha256 = unsigned.pop("sha256", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    actual_sha256 = hashlib.sha256(encoded).hexdigest()
    if expected_sha256 != actual_sha256:
        raise ValueError(
            f"Manifest SHA256 mismatch: stored={expected_sha256}, computed={actual_sha256}"
        )
    for batch in manifest.get("batches", []):
        indices = batch.get("dataset_indices", [])
        expected = batch.get("samples", [])
        actual = [frame_identity(dataset, index) for index in indices]
        if actual != expected:
            raise ValueError(
                f"Manifest metadata mismatch in batch {batch.get('batch_index')}: "
                f"expected={expected}, actual={actual}"
            )


def seam_map_diagnostics(pair_u, pair_v, pair_valid):
    if pair_u is None or pair_v is None or pair_valid is None:
        raise RuntimeError("Seam map diagnostics require pair_u, pair_v, and pair_valid")
    pair_u = pair_u.detach().cpu()
    pair_v = pair_v.detach().cpu()
    valid = pair_valid.detach().cpu() > 0
    _, _, height, width = valid.shape
    source = torch.nonzero(valid[0, 0], as_tuple=False)
    if not source.numel():
        return {"atlas_pair_texels": 0, "atlas_total_texels": height * width}
    source_y, source_x = source[:, 0], source[:, 1]
    target_x_float = (pair_u[0, 0, source_y, source_x] + 1.0) * 0.5 * max(width - 1, 1)
    target_y_float = (pair_v[0, 0, source_y, source_x] + 1.0) * 0.5 * max(height - 1, 1)
    target_x = target_x_float.round().long()
    target_y = target_y_float.round().long()
    in_bounds = (
        (target_x >= 0) & (target_x < width) & (target_y >= 0) & (target_y < height)
    )
    center_error = torch.maximum(
        (target_x_float - target_x.float()).abs(),
        (target_y_float - target_y.float()).abs(),
    )
    safe_x = target_x.clamp(0, width - 1)
    safe_y = target_y.clamp(0, height - 1)
    target_is_mapped = valid[0, 0, safe_y, safe_x] & in_bounds
    back_x = ((pair_u[0, 0, safe_y, safe_x] + 1.0) * 0.5 * max(width - 1, 1)).round().long()
    back_y = ((pair_v[0, 0, safe_y, safe_x] + 1.0) * 0.5 * max(height - 1, 1)).round().long()
    reciprocal = target_is_mapped & (back_x == source_x) & (back_y == source_y)
    target_linear = safe_y * width + safe_x
    target_counts = torch.bincount(target_linear, minlength=height * width)
    return {
        "atlas_pair_texels": int(source.shape[0]),
        "atlas_total_texels": height * width,
        "out_of_range_targets": int((~in_bounds).sum()),
        "off_center_targets": int((center_error > 1e-4).sum()),
        "max_center_error_px": float(center_error.max()),
        "self_pairs": int(((target_x == source_x) & (target_y == source_y)).sum()),
        "target_mapped_texels": int(target_is_mapped.sum()),
        "reciprocal_pairs": int(reciprocal.sum()),
        "nonreciprocal_pairs": int((~reciprocal).sum()),
        "duplicate_target_texels": int((target_counts > 1).sum()),
        "maximum_sources_per_target": int(target_counts.max()),
    }


def uv_bake_assignment_diagnostics(outputs, face_dec, uv_visibility):
    """Count UV splat coverage/collisions without changing the baker or its masks."""
    batch, _, uv_h, uv_w = uv_visibility.shape
    face_id = outputs["face_id"]
    face_bw = outputs["face_bw"]
    image_valid = outputs["valid_nograd"].bool() & (face_id >= 0)
    faces_uvs = face_dec.faces_uvs.to(face_id.device)
    vertex_uvs = face_dec.vertex_uvs.to(face_bw.device)
    geometry_masks = []
    stats = []
    for sample_index in range(batch):
        ys, xs = torch.where(image_valid[sample_index])
        count = torch.zeros(uv_h * uv_w, dtype=torch.long, device=face_id.device)
        out_of_range = 0
        quantization_mean = 0.0
        quantization_max = 0.0
        if ys.numel():
            triangles = face_id[sample_index, ys, xs].long()
            tri_uv = vertex_uvs[faces_uvs[triangles]]
            bary = face_bw[sample_index, ys, xs]
            if bary.shape[-1] == 2:
                bary = torch.cat((bary, 1.0 - bary.sum(dim=-1, keepdim=True)), dim=-1)
            uv = (tri_uv * bary[..., None]).sum(dim=1)
            out_of_range = int(((uv < 0) | (uv > 1)).any(dim=1).sum().detach())
            px = torch.stack((uv[:, 0] * (uv_w - 1), uv[:, 1] * (uv_h - 1)), dim=1)
            texel = px.long()
            quantization = (px - texel).abs().amax(dim=1)
            quantization_mean = float(quantization.mean().detach())
            quantization_max = float(quantization.max().detach())
            uv_x = texel[:, 0].clamp(0, uv_w - 1)
            uv_y = texel[:, 1].clamp(0, uv_h - 1)
            linear = uv_y * uv_w + uv_x
            count = torch.bincount(linear, minlength=uv_h * uv_w)
        geometry_mask = (count > 0).view(1, uv_h, uv_w)
        final_mask = uv_visibility[sample_index].detach() > 0
        stats.append({
            "rasterized_face_pixels": int(ys.numel()),
            "unique_baked_texels": int((count > 0).sum()),
            "collision_texels": int((count > 1).sum()),
            "collision_writes": int((count - 1).clamp_min(0).sum()),
            "maximum_writes_per_texel": int(count.max()) if count.numel() else 0,
            "out_of_range_uv_pixels": out_of_range,
            "mean_floor_quantization_error_px": quantization_mean,
            "max_floor_quantization_error_px": quantization_max,
            "confidence_mask_filtered_texels": int((geometry_mask & ~final_mask).sum()),
            "unexpected_final_visibility_texels": int((~geometry_mask & final_mask).sum()),
        })
        geometry_masks.append(geometry_mask)
    return torch.stack(geometry_masks), stats


def trainable_parameters(trainer):
    result = []
    seen = set()
    for group in trainer.optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.requires_grad and id(parameter) not in seen:
                result.append(parameter)
                seen.add(id(parameter))
    return result


def gradient_statistics(loss, parameters, retain_graph=True):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    l1_sum = 0.0
    squared_sum = 0.0
    max_abs = 0.0
    tensor_count = 0
    element_count = 0
    finite = True
    for gradient in gradients:
        if gradient is None:
            continue
        detached = gradient.detach()
        tensor_count += 1
        element_count += detached.numel()
        finite = finite and bool(torch.isfinite(detached).all())
        absolute = detached.abs()
        l1_sum += float(absolute.sum())
        squared_sum += float(detached.double().square().sum())
        if detached.numel():
            max_abs = max(max_abs, float(absolute.max()))
    return {
        "l2": math.sqrt(squared_sum), "l1": l1_sum, "max_abs": max_abs,
        "finite": finite, "tensor_count": tensor_count, "element_count": element_count,
    }


def gradient_norm(loss, parameters, retain_graph=True):
    return gradient_statistics(loss, parameters, retain_graph=retain_graph)["l2"]


def accumulated_gradient_statistics(parameters):
    l1_sum = 0.0
    squared_sum = 0.0
    max_abs = 0.0
    tensor_count = 0
    element_count = 0
    finite = True
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        tensor_count += 1
        element_count += gradient.numel()
        finite = finite and bool(torch.isfinite(gradient).all())
        absolute = gradient.abs()
        l1_sum += float(absolute.sum())
        squared_sum += float(gradient.double().square().sum())
        if gradient.numel():
            max_abs = max(max_abs, float(absolute.max()))
    return {
        "l2": math.sqrt(squared_sum), "l1": l1_sum, "max_abs": max_abs,
        "finite": finite, "tensor_count": tensor_count, "element_count": element_count,
    }


def report(name, status, **details):
    global REPORT_HANDLE
    payload = {"test": name, "status": status, **details}
    line = json.dumps(payload, sort_keys=True)
    print(line)
    if REPORT_HANDLE is not None:
        REPORT_HANDLE.write(line + "\n")
        REPORT_HANDLE.flush()
    return status == "FAIL"


def seam_visibility_diagnostics(uv_mask, pair_u, pair_v, pair_valid, band_px):
    """Reproduce only the Seam masks to diagnose supervision availability."""
    if pair_u is None or pair_v is None or pair_valid is None:
        raise RuntimeError("Seam diagnostics require pair_u, pair_v, and pair_valid")
    if uv_mask.ndim != 4 or uv_mask.shape[1] != 1:
        raise ValueError(f"Expected UV visibility in B1HW layout, got {tuple(uv_mask.shape)}")

    batch, _, height, width = uv_mask.shape
    device, dtype = uv_mask.device, uv_mask.dtype
    pair_u = pair_u.to(device=device, dtype=dtype)
    pair_v = pair_v.to(device=device, dtype=dtype)
    pair_valid = pair_valid.to(device=device, dtype=dtype)
    if pair_valid.shape != (1, 1, height, width):
        raise ValueError(
            f"Seam pair_valid must be [1, 1, {height}, {width}], got {tuple(pair_valid.shape)}"
        )

    visibility = uv_mask.detach().clamp(0, 1)
    grid = torch.stack((pair_u, pair_v), dim=-1).expand(batch, -1, -1, -1, -1).squeeze(1)
    partner_visibility = F.grid_sample(
        visibility, grid, mode="nearest", padding_mode="zeros", align_corners=True
    )
    candidates = pair_valid.expand(batch, -1, -1, -1) > 0
    source_visible = candidates & (visibility > 0)
    partner_visible = candidates & (partner_visibility > 0)
    mutual_visible = source_visible & partner_visible

    valid_source = pair_valid[0, 0] > 0
    source_y, source_x = torch.where(valid_source)
    reciprocal_map = torch.zeros_like(valid_source)
    duplicate_target_map = torch.zeros_like(valid_source)
    source_boundary_map = torch.zeros_like(valid_source)
    partner_boundary_map = torch.zeros_like(valid_source)
    if source_y.numel():
        target_x = ((pair_u[0, 0, source_y, source_x] + 1.0) * 0.5 * max(width - 1, 1)).round().long()
        target_y = ((pair_v[0, 0, source_y, source_x] + 1.0) * 0.5 * max(height - 1, 1)).round().long()
        target_x = target_x.clamp(0, width - 1)
        target_y = target_y.clamp(0, height - 1)
        back_x = ((pair_u[0, 0, target_y, target_x] + 1.0) * 0.5 * max(width - 1, 1)).round().long()
        back_y = ((pair_v[0, 0, target_y, target_x] + 1.0) * 0.5 * max(height - 1, 1)).round().long()
        reciprocal_map[source_y, source_x] = (
            (pair_valid[0, 0, target_y, target_x] > 0)
            & (back_x == source_x) & (back_y == source_y)
        )
        target_linear = target_y * width + target_x
        target_counts = torch.bincount(target_linear, minlength=height * width)
        duplicate_target_map[source_y, source_x] = target_counts[target_linear] > 1
        source_boundary_map[source_y, source_x] = (
            (source_x == 0) | (source_x == width - 1)
            | (source_y == 0) | (source_y == height - 1)
        )
        partner_boundary_map[source_y, source_x] = (
            (target_x == 0) | (target_x == width - 1)
            | (target_y == 0) | (target_y == height - 1)
        )

    reciprocal = reciprocal_map[None, None].expand(batch, -1, -1, -1)
    duplicate_target = duplicate_target_map[None, None].expand(batch, -1, -1, -1)
    source_boundary = source_boundary_map[None, None].expand(batch, -1, -1, -1)
    partner_boundary = partner_boundary_map[None, None].expand(batch, -1, -1, -1)

    kernel = 2 * int(band_px) + 1
    band = F.max_pool2d(pair_valid, kernel, stride=1, padding=int(band_px)) > 0
    effective = band.expand(batch, -1, -1, -1) & (visibility > 0)
    tv_edges_x = effective[..., 1:] & effective[..., :-1]
    tv_edges_y = effective[..., 1:, :] & effective[..., :-1, :]

    def per_batch_count(mask):
        return [int(value) for value in mask.flatten(1).sum(dim=1).detach().cpu().tolist()]

    return {
        "atlas_pair_texels": int((pair_valid > 0).sum().detach()),
        "atlas_total_texels": height * width,
        "candidate_pair_texels": per_batch_count(candidates),
        "source_visible_pair_texels": per_batch_count(source_visible),
        "partner_visible_pair_texels": per_batch_count(partner_visible),
        "mutual_pair_texels": per_batch_count(mutual_visible),
        "reciprocal_candidate_pair_texels": per_batch_count(candidates & reciprocal),
        "nonreciprocal_candidate_pair_texels": per_batch_count(candidates & ~reciprocal),
        "reciprocal_mutual_pair_texels": per_batch_count(mutual_visible & reciprocal),
        "nonreciprocal_mutual_pair_texels": per_batch_count(mutual_visible & ~reciprocal),
        "duplicate_target_candidate_pair_texels": per_batch_count(candidates & duplicate_target),
        "duplicate_target_mutual_pair_texels": per_batch_count(mutual_visible & duplicate_target),
        "source_boundary_candidate_pair_texels": per_batch_count(candidates & source_boundary),
        "source_boundary_mutual_pair_texels": per_batch_count(mutual_visible & source_boundary),
        "partner_boundary_candidate_pair_texels": per_batch_count(candidates & partner_boundary),
        "partner_boundary_mutual_pair_texels": per_batch_count(mutual_visible & partner_boundary),
        "tv_effective_texels": per_batch_count(effective),
        "tv_effective_edges": [
            x + y for x, y in zip(per_batch_count(tv_edges_x), per_batch_count(tv_edges_y))
        ],
        "source_hits": source_visible.sum(dim=0).detach().cpu(),
        "mutual_hits": mutual_visible.sum(dim=0).detach().cpu(),
    }


def main():
    global REPORT_HANDLE
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument(
        "--sampler-mode", choices=("auto", "paired", "sequential"), default="auto",
        help="Smoke-test sampling only; does not change the training DataLoader",
    )
    parser.add_argument("--manifest-in", default=None)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--jsonl-out", default=None)
    parser.add_argument(
        "--losses", default="all",
        help="Comma-separated subset of lf,hf,uv,seam, or 'all' (default: all)",
    )
    args = parser.parse_args()
    if args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    if args.manifest_in and args.manifest_out:
        parser.error("Use only one of --manifest-in and --manifest-out")
    if args.jsonl_out:
        output_path = Path(args.jsonl_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        REPORT_HANDLE = output_path.open("w")

    selected = {name.strip().lower() for name in args.losses.split(",") if name.strip()}
    if "all" in selected:
        selected = {"lf", "hf", "uv", "seam"}
    unknown = selected - {"lf", "hf", "uv", "seam"}
    if unknown:
        parser.error(f"Unknown losses: {sorted(unknown)}")

    failed = False
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser()
        if not checkpoint_path.is_file():
            failed |= report(
                "checkpoint_input", "FAIL", initialization_mode="checkpoint",
                checkpoint=str(checkpoint_path), restore_verified=False,
                reason="checkpoint_path_does_not_exist_or_is_not_a_file",
            )
            if REPORT_HANDLE is not None:
                REPORT_HANDLE.close()
                REPORT_HANDLE = None
            raise SystemExit(1)
        args.checkpoint = str(checkpoint_path.resolve())
        initialization_mode = "checkpoint"
        failed |= report(
            "checkpoint_input", "PASS", initialization_mode=initialization_mode,
            checkpoint=args.checkpoint, checkpoint_bytes=checkpoint_path.stat().st_size,
            restore_verified=False, reason="restore_pending",
        )
    else:
        initialization_mode = "random_initialization"
        report(
            "checkpoint_input", "WARN", initialization_mode=initialization_mode,
            checkpoint=None, restore_verified=False,
            reason="no_checkpoint_requested_seeded_random_initialization",
        )

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    overrides = {
        "training.gpus": str(args.gpu),
        "data.num_workers": 0,
        "data.per_gpu_batch_size": args.batch_size,
        "exp.lf_uv.every": 1,
        "exp.lf_img.enable": "lf" in selected,
        "exp.hf_img.enable": "hf" in selected,
        "exp.lf_uv.enable": "uv" in selected,
        "exp.lf_uv.loss.enable": "uv" in selected,
        "exp.lf_uv.seam.enable": "seam" in selected,
    }
    if args.checkpoint:
        overrides["training.pretrained_checkpoint_path"] = args.checkpoint
    config.update(overrides)
    normalize_dotted_config(config, overrides)
    nonzero_defaults = {
        "exp.lf_img.lambda": 0.1,
        "exp.hf_img.lambda": 0.1,
        "exp.lf_uv.loss.weight": 0.05,
    }
    selected_keys = {"lf": "exp.lf_img.lambda", "hf": "exp.hf_img.lambda", "uv": "exp.lf_uv.loss.weight"}
    forced_weights = {
        key: nonzero_defaults[key]
        for name, key in selected_keys.items()
        if name in selected and float(config.get(key, 0.0)) <= 0.0
    }
    if forced_weights:
        config.update(forced_weights)
        normalize_dotted_config(config, forced_weights)
    hf_cfg = validate_hf_image_config(config, context="Smoke-test HF-IMG") if "hf" in selected else None
    config["bald"] = Path(args.config).parent.name == "218"
    config["tb_writer"] = None
    config["datatype"] = "nersemble"

    logger = logging.getLogger("fresco-smoke")
    logger.handlers = [logging.StreamHandler(sys.stdout)]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    config["logger"] = logger

    set_reproducible_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    with tempfile.TemporaryDirectory(prefix="fresco_smoke_") as workspace:
        config["local_workspace"] = workspace
        config["exp.lf_uv.seam.cache_dir"] = os.path.join(workspace, "uv_cache")
        dataset = NeRSembleData(config)
        required_frame_fields = {
            "timestep_index", "camera_index", "camera_id", "flame_param_path",
            "file_path", "transform_matrix",
        }
        missing_frame_fields = required_frame_fields - set(dataset.framelist[0])
        failed |= report(
            "dataset_metadata", "PASS" if not missing_frame_fields else "FAIL",
            samples=len(dataset), missing_fields=sorted(missing_frame_fields),
            timestep_type=type(dataset.framelist[0].get("timestep_index")).__name__,
            camera_id_type=type(dataset.framelist[0].get("camera_id")).__name__,
            camera_index_type=type(dataset.framelist[0].get("camera_index")).__name__,
        )
        batch_size = int(config["data.per_gpu_batch_size"])
        uv_enabled = (
            config.get("exp.lf_uv.enable", False)
            and config.get("exp.lf_uv.loss.enable", False)
            and float(config.get("exp.lf_uv.loss.weight", 0.0)) > 0.0
        )
        requested_mode = args.sampler_mode
        sampler_mode = (
            "paired" if requested_mode == "paired" or (requested_mode == "auto" and uv_enabled)
            else "sequential"
        )
        if args.manifest_in:
            with open(args.manifest_in, "r") as stream:
                manifest = json.load(stream)
            validate_batch_manifest(manifest, dataset, batch_size)
            if int(manifest.get("seed", -1)) != args.seed:
                raise ValueError(
                    f"Manifest seed={manifest.get('seed')} does not match --seed={args.seed}"
                )
            if requested_mode != "auto" and manifest["sampler_mode"] != requested_mode:
                raise ValueError(
                    f"Manifest sampler_mode={manifest['sampler_mode']} does not match "
                    f"--sampler-mode={requested_mode}"
                )
            sampler_mode = manifest["sampler_mode"]
            if len(manifest["batches"]) < args.max_batches:
                raise ValueError(
                    f"Manifest has {len(manifest['batches'])} batches but "
                    f"--max-batches={args.max_batches}"
                )
            manifest["batches"] = manifest["batches"][:args.max_batches]
        else:
            if sampler_mode == "paired":
                source_sampler = SameFramePairBatchSampler(
                    dataset, batch_size, shuffle=True, drop_last=False
                )
                batches = list(itertools.islice(iter(source_sampler), args.max_batches))
                failed |= report(
                    "sampler", "PASS", mode=sampler_mode,
                    source_images=source_sampler.num_source_samples,
                    emitted_images=source_sampler.num_emitted_samples,
                    sample_delta=source_sampler.epoch_sample_delta,
                    excluded_single_view=source_sampler.num_excluded_single_view_samples,
                    available_batches=len(source_sampler), selected_batches=len(batches),
                )
            else:
                batches = [
                    list(range(start, min(start + batch_size, len(dataset))))
                    for start in range(0, len(dataset), batch_size)
                ][:args.max_batches]
                failed |= report(
                    "sampler", "PASS", mode=sampler_mode,
                    available_batches=(len(dataset) + batch_size - 1) // batch_size,
                    selected_batches=len(batches),
                )
            manifest = make_batch_manifest(dataset, batches, sampler_mode, batch_size, args.seed)
            if args.manifest_out:
                manifest_path = Path(args.manifest_out)
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                with manifest_path.open("w") as stream:
                    json.dump(manifest, stream, indent=2, sort_keys=True)

        manifest_batches = [batch["dataset_indices"] for batch in manifest["batches"]]
        loader = DataLoader(
            dataset, batch_sampler=ManifestBatchSampler(manifest_batches), num_workers=0
        )
        failed |= report(
            "batch_manifest", "PASS", sampler_mode=sampler_mode,
            manifest_sha256=manifest["sha256"], manifest_input=args.manifest_in,
            manifest_output=args.manifest_out, batches=len(manifest_batches),
            batch_size=batch_size, seed=args.seed,
        )
        failed |= report(
            "run_configuration", "PASS", selected_losses=sorted(selected),
            checkpoint=args.checkpoint, sampler_mode=sampler_mode, seed=args.seed,
            initialization_mode=initialization_mode,
            manifest_sha256=manifest["sha256"], max_batches=args.max_batches,
        )

        # Manifest construction may consume Python RNG. Reset all RNGs so a
        # generated manifest and a replayed manifest initialize identical models.
        set_reproducible_seed(args.seed)

        trainer = JointTrainer(
            config, logger, dataset.radius, all_flame_params=dataset.load_all_flame_params()
        )
        optimizer_state_entries = len(trainer.optimizer.state)
        resume_ok = bool(args.checkpoint) and optimizer_state_entries > 0
        failed |= report(
            "resume_state", "PASS" if resume_ok else "WARN" if not args.checkpoint else "FAIL",
            initialization_mode=initialization_mode, restore_verified=resume_ok,
            reason=("" if resume_ok else "seeded_random_initialization" if not args.checkpoint
                    else "checkpoint_loaded_without_optimizer_state"),
            current_epoch=trainer.current_epoch,
            global_step=trainer.global_step,
            learning_rates=[group["lr"] for group in trainer.optimizer.param_groups],
            optimizer_state_entries=optimizer_state_entries, checkpoint=args.checkpoint,
        )
        optimizer_parameter_ids = {
            id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]
        }
        optimized_flame = [
            value for value in trainer.all_flame_params.values() if value.requires_grad
        ]
        flame_binding_ok = all(id(value) in optimizer_parameter_ids for value in optimized_flame)
        failed |= report(
            "flame_optimizer_binding", "PASS" if flame_binding_ok else "FAIL",
            optimized_tensors=len(optimized_flame),
            optimizer_bound_tensors=sum(id(value) in optimizer_parameter_ids for value in optimized_flame),
        )
        trainer.set_train()
        parameters = trainable_parameters(trainer)
        # Keep stochastic forward behavior aligned across Seam-only and All,
        # even if their diagnostic setup consumed a different number of RNG calls.
        set_reproducible_seed(args.seed)
        face_dec = trainer.facewrapper.flame_dec
        mapping_stats = seam_map_diagnostics(
            trainer.seam_maps.get("pair_u"), trainer.seam_maps.get("pair_v"),
            trainer.seam_maps.get("pair_valid"),
        ) if "seam" in selected else None
        if mapping_stats is not None:
            faces_np = face_dec.faces.detach().cpu().numpy()
            faces_uvs_np = face_dec.faces_uvs.detach().cpu().numpy()
            vertex_uvs_np = face_dec.vertex_uvs.detach().cpu().numpy()
            seam_edges = _detect_seam_edges(
                _build_edge_face_adjacency(faces_np), faces_np, faces_uvs_np, vertex_uvs_np
            )
            mapping_status = (
                "FAIL" if mapping_stats.get("out_of_range_targets", 0)
                or mapping_stats.get("off_center_targets", 0)
                else "WARN" if mapping_stats.get("nonreciprocal_pairs", 0) else "PASS"
            )
            failed |= report(
                "seam_atlas_mapping", mapping_status, **mapping_stats,
                topology_seam_edges=len(seam_edges),
                topology_correspondence="shared_3d_vertex_ids_with_matched_edge_parameter",
            )
        seam_summary = {
            "batches": 0,
            "samples": 0,
            "pair_valid_batches": 0,
            "pair_valid_samples": 0,
            "tv_valid_batches": 0,
            "tv_valid_samples": 0,
            "candidate_pair_texels": 0,
            "source_visible_pair_texels": 0,
            "partner_visible_pair_texels": 0,
            "mutual_pair_texels": 0,
            "tv_effective_texels": 0,
            "tv_effective_edges": 0,
            "pair_losses": [],
            "tv_losses": [],
            "pair_gradients": [],
            "tv_gradients": [],
            "weighted_pair_gradients": [],
            "weighted_tv_gradients": [],
            "per_batch_mutual_pair_texels": [],
            "uv_bake_assignment": [],
            "source_hits": None,
            "mutual_hits": None,
        }

        for batch_index, items in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            trainer.global_step += 1
            trainer.stage_step += 1
            trainer.set_data(items)
            manifest_batch = manifest["batches"][batch_index]
            dataset_indices = manifest_batch["dataset_indices"]
            expected_timesteps = [sample["timestep"] for sample in manifest_batch["samples"]]
            expected_cameras = [sample["camera_id"] for sample in manifest_batch["samples"]]
            actual_timesteps = [int(x) for x in items["frame_idx"].tolist()]
            actual_cameras = [str(x) for x in items["cam"]]
            identity_ok = actual_timesteps == expected_timesteps and actual_cameras == expected_cameras
            batch_fields_ok = (
                trainer.img.shape[0] == len(items["frame_idx"])
                and trainer.img.shape[-1] == 3
                and len(items["cam"]) == len(items["frame_idx"])
            )
            failed |= report(
                "dataset_getitem_and_collate", "PASS" if batch_fields_ok else "FAIL",
                batch_index=batch_index, dataset_indices=dataset_indices,
                timesteps=actual_timesteps, camera_ids=actual_cameras,
                keys=sorted(items), image_shape=list(items["img"].shape),
                frame_shape=list(items["frame_idx"].shape), batch_size=len(items["frame_idx"]),
            )
            failed |= report(
                "manifest_replay", "PASS" if identity_ok else "FAIL", batch_index=batch_index,
                dataset_indices=dataset_indices, expected_timesteps=expected_timesteps,
                actual_timesteps=actual_timesteps, expected_camera_ids=expected_cameras,
                actual_camera_ids=actual_cameras, manifest_sha256=manifest["sha256"],
            )
            outputs = trainer.network_forward()
            trainer._prepare_cross_view_uv(outputs)

            confidence = outputs.get("view_confidence")
            valid = outputs.get("valid_nograd")
            confidence_ok = (
                confidence is not None and torch.isfinite(confidence).all()
                and float(confidence.min()) >= 0.0 and float(confidence.max()) <= 1.0
                and valid is not None and float(confidence[valid].mean()) > 0.0
            )
            failed |= report(
                "camera_space_confidence", "PASS" if confidence_ok else "FAIL",
                minimum=float(confidence.min()) if confidence is not None else None,
                maximum=float(confidence.max()) if confidence is not None else None,
                valid_mean=float(confidence[valid].mean()) if confidence is not None and valid.any() else 0.0,
            )

            if "lf" in selected:
                lf_cfg = {"kernel": config.get("exp.lf_img.kernel", 9),
                          "sigma": config.get("exp.lf_img.sigma", -1.0)}
                lf = compute_lfimg_loss(outputs["render_fuse"], trainer.img, None, lf_cfg)["loss"]
                lf_grad = gradient_norm(lf, parameters)
                failed |= report("lf", "PASS" if torch.isfinite(lf) and lf_grad > 0 else "FAIL",
                                 loss=float(lf.detach()), gradient_norm=lf_grad)
            else:
                report("lf", "SKIP", reason="not_selected")

            if "hf" in selected:
                hf_stats = compute_hfimg_loss(outputs["render_fuse"], trainer.img, None, hf_cfg)
                hf_grad = gradient_norm(hf_stats["loss"], parameters)
                failed |= report(
                    "hf", "PASS" if torch.isfinite(hf_stats["loss"]) and hf_grad > 0 else "FAIL",
                    loss=float(hf_stats["loss"].detach()), edge=float(hf_stats["edge_hf"]),
                    gdl=float(hf_stats["gdl"]), gradient_norm=hf_grad,
                    parameter_status=hf_stats["parameter_status"],
                )
            else:
                report("hf", "SKIP", reason="not_selected")

            if uv_enabled:
                pairs = build_same_frame_pairs(trainer.frame_idx, trainer.camera_ids)
                geometry_max_diff = 0.0
                for a, b in pairs:
                    for value in trainer.flame_params.values():
                        geometry_max_diff = max(
                            geometry_max_diff, float((value[a] - value[b]).detach().abs().max())
                        )
                failed |= report(
                    "real_batch_pairing", "PASS" if pairs and geometry_max_diff == 0.0 else "FAIL", pairs=pairs,
                    frame_indices=[int(x) for x in trainer.frame_idx.cpu().tolist()],
                    camera_ids=[str(x) for x in trainer.camera_ids],
                    flame_geometry_max_diff=geometry_max_diff,
                )
                uv_stats = compute_cross_view_uv_loss(
                    outputs["__uv_render__"], outputs["__uv_visibility__"],
                    outputs["__uv_confidence__"], trainer.frame_idx, trainer.camera_ids,
                )
                uv_grad = gradient_norm(uv_stats["loss"], parameters)
                uv_ok = not uv_stats["skipped"] and uv_stats["coverage"] > 0 and uv_grad > 0
                failed |= report(
                    "cross_view_uv", "PASS" if uv_ok else "FAIL",
                    loss=float(uv_stats["loss"].detach()), coverage=uv_stats["coverage"],
                    mean_confidence=uv_stats["mean_confidence"], pairs=uv_stats["pair_count"],
                    mutual_texels=uv_stats["mutual_texels"],
                    gradient_norm=uv_grad, reason=uv_stats["reason"],
                )
            else:
                report("cross_view_uv", "SKIP", reason="disabled_by_config")

            if "seam" in selected:
                seam_cfg = {
                    "weight_pair": config.get("exp.lf_uv.seam.weight_pair", 0.02),
                    "weight_tv": config.get("exp.lf_uv.seam.weight_tv", 0.004),
                    "band_px": config.get("exp.lf_uv.seam.band_px", 2),
                }
                seam = compute_seam_loss(
                    outputs["__uv_render__"], seam_cfg, uv_mask=outputs["__uv_visibility__"],
                    pair_u=trainer.seam_maps.get("pair_u"), pair_v=trainer.seam_maps.get("pair_v"),
                    pair_valid=trainer.seam_maps.get("pair_valid"),
                )
                pair_probe_cfg = dict(seam_cfg, weight_pair=1.0, weight_tv=0.0)
                tv_probe_cfg = dict(seam_cfg, weight_pair=0.0, weight_tv=1.0)
                seam_kwargs = {
                    "uv_mask": outputs["__uv_visibility__"],
                    "pair_u": trainer.seam_maps.get("pair_u"),
                    "pair_v": trainer.seam_maps.get("pair_v"),
                    "pair_valid": trainer.seam_maps.get("pair_valid"),
                }
                pair_probe = compute_seam_loss(
                    outputs["__uv_render__"], pair_probe_cfg, **seam_kwargs
                )
                tv_probe = compute_seam_loss(
                    outputs["__uv_render__"], tv_probe_cfg, **seam_kwargs
                )
                visibility_stats = seam_visibility_diagnostics(
                    outputs["__uv_visibility__"],
                    trainer.seam_maps.get("pair_u"),
                    trainer.seam_maps.get("pair_v"),
                    trainer.seam_maps.get("pair_valid"),
                    seam_cfg["band_px"],
                )
                geometry_visibility, bake_assignment = uv_bake_assignment_diagnostics(
                    outputs, face_dec, outputs["__uv_visibility__"]
                )
                geometry_visibility_stats = seam_visibility_diagnostics(
                    geometry_visibility.to(
                        device=outputs["__uv_visibility__"].device,
                        dtype=outputs["__uv_visibility__"].dtype,
                    ),
                    trainer.seam_maps.get("pair_u"),
                    trainer.seam_maps.get("pair_v"),
                    trainer.seam_maps.get("pair_valid"),
                    seam_cfg["band_px"],
                )

                pair_loss = pair_probe["loss"]
                tv_loss = tv_probe["loss"]
                pair_gradient = gradient_statistics(pair_loss, parameters)
                tv_gradient = gradient_statistics(tv_loss, parameters)
                seam_gradient = gradient_statistics(seam["loss"], parameters)
                pair_grad = pair_gradient["l2"]
                tv_grad = tv_gradient["l2"]
                seam_grad = seam_gradient["l2"]
                mutual_count = sum(visibility_stats["mutual_pair_texels"])
                tv_edge_count = sum(visibility_stats["tv_effective_edges"])
                pair_matches = torch.isclose(pair_loss.detach(), seam["pair"], rtol=1e-5, atol=1e-7)
                tv_matches = torch.isclose(tv_loss.detach(), seam["tv"], rtol=1e-5, atol=1e-7)
                pair_value = float(pair_loss.detach())
                tv_value = float(tv_loss.detach())

                if mutual_count == 0:
                    pair_status = "SKIP" if pair_value == 0.0 and pair_grad == 0.0 else "FAIL"
                    pair_reason = "no_mutual_visibility"
                elif not torch.isfinite(pair_loss) or not pair_matches or not pair_gradient["finite"]:
                    pair_status = "FAIL"
                    pair_reason = "pair_interface_value_mismatch_or_nonfinite"
                elif pair_value == 0.0 and pair_grad == 0.0:
                    pair_status = "WARN"
                    pair_reason = "valid_texels_but_predictions_identical"
                elif pair_grad <= 0:
                    pair_status = "FAIL"
                    pair_reason = "valid_nonzero_pair_loss_but_no_gradient"
                else:
                    pair_status = "PASS"
                    pair_reason = ""
                if tv_edge_count == 0:
                    tv_status = "SKIP" if tv_value == 0.0 and tv_grad == 0.0 else "FAIL"
                    tv_reason = "no_visible_tv_edges"
                elif not torch.isfinite(tv_loss) or not tv_matches or not tv_gradient["finite"]:
                    tv_status = "FAIL"
                    tv_reason = "tv_interface_value_mismatch_or_nonfinite"
                elif tv_value == 0.0 and tv_grad == 0.0:
                    tv_status = "WARN"
                    tv_reason = "valid_tv_edges_but_texture_is_locally_constant"
                elif tv_grad <= 0:
                    tv_status = "FAIL"
                    tv_reason = "valid_nonzero_tv_loss_but_no_gradient"
                else:
                    tv_status = "PASS"
                    tv_reason = ""

                failed |= report(
                    "uv_bake_assignment", "PASS" if all(
                        stat["unexpected_final_visibility_texels"] == 0 for stat in bake_assignment
                    ) else "FAIL",
                    batch_index=batch_index, dataset_indices=dataset_indices,
                    samples=bake_assignment,
                    geometry_source_visible_pair_texels=geometry_visibility_stats[
                        "source_visible_pair_texels"
                    ],
                    geometry_partner_visible_pair_texels=geometry_visibility_stats[
                        "partner_visible_pair_texels"
                    ],
                    geometry_mutual_pair_texels=geometry_visibility_stats["mutual_pair_texels"],
                    final_source_visible_pair_texels=visibility_stats["source_visible_pair_texels"],
                    final_partner_visible_pair_texels=visibility_stats["partner_visible_pair_texels"],
                    final_mutual_pair_texels=visibility_stats["mutual_pair_texels"],
                )

                failed |= report(
                    "seam_pair", pair_status, batch_index=batch_index,
                    dataset_indices=dataset_indices, timesteps=actual_timesteps,
                    loss=pair_value, gradient_norm=pair_grad,
                    gradient_l1_sum=pair_gradient["l1"],
                    gradient_max_abs=pair_gradient["max_abs"],
                    gradient_finite=pair_gradient["finite"],
                    gradient_tensor_count=pair_gradient["tensor_count"],
                    gradient_element_count=pair_gradient["element_count"],
                    weighted_gradient_norm=abs(float(seam_cfg["weight_pair"])) * pair_grad,
                    loss_denominator=max(mutual_count * outputs["__uv_render__"].shape[1], 1),
                    atlas_pair_texels=visibility_stats["atlas_pair_texels"],
                    candidate_pair_texels=visibility_stats["candidate_pair_texels"],
                    source_visible_pair_texels=visibility_stats["source_visible_pair_texels"],
                    partner_visible_pair_texels=visibility_stats["partner_visible_pair_texels"],
                    mutual_pair_texels=visibility_stats["mutual_pair_texels"],
                    reciprocal_mutual_pair_texels=visibility_stats[
                        "reciprocal_mutual_pair_texels"
                    ],
                    nonreciprocal_mutual_pair_texels=visibility_stats[
                        "nonreciprocal_mutual_pair_texels"
                    ],
                    duplicate_target_mutual_pair_texels=visibility_stats[
                        "duplicate_target_mutual_pair_texels"
                    ],
                    source_boundary_mutual_pair_texels=visibility_stats[
                        "source_boundary_mutual_pair_texels"
                    ],
                    partner_boundary_mutual_pair_texels=visibility_stats[
                        "partner_boundary_mutual_pair_texels"
                    ],
                    frame_indices=actual_timesteps, camera_ids=actual_cameras,
                    interface_value_matches=bool(pair_matches), reason=pair_reason,
                )
                failed |= report(
                    "seam_tv", tv_status, batch_index=batch_index,
                    dataset_indices=dataset_indices, timesteps=actual_timesteps,
                    camera_ids=actual_cameras, loss=tv_value, gradient_norm=tv_grad,
                    gradient_l1_sum=tv_gradient["l1"],
                    gradient_max_abs=tv_gradient["max_abs"],
                    gradient_finite=tv_gradient["finite"],
                    gradient_tensor_count=tv_gradient["tensor_count"],
                    gradient_element_count=tv_gradient["element_count"],
                    weighted_gradient_norm=abs(float(seam_cfg["weight_tv"])) * tv_grad,
                    effective_texels=visibility_stats["tv_effective_texels"],
                    effective_edges=visibility_stats["tv_effective_edges"],
                    interface_value_matches=bool(tv_matches), reason=tv_reason,
                )
                seam_ok = (
                    not seam["skipped"] and torch.isfinite(seam["loss"])
                    and (seam_grad > 0 or (mutual_count == 0 and tv_edge_count == 0))
                )
                failed |= report(
                    "seam", "PASS" if seam_ok else "FAIL", batch_index=batch_index,
                    dataset_indices=dataset_indices, timesteps=actual_timesteps,
                    camera_ids=actual_cameras,
                    loss=float(seam["loss"].detach()),
                    pair=float(seam["pair"]), tv=float(seam["tv"]),
                    coverage=seam["effective_cov"], pair_coverage=seam["pair_cov"],
                    pair_gradient_norm=pair_grad, tv_gradient_norm=tv_grad,
                    pair_weighted_gradient_norm=abs(float(seam_cfg["weight_pair"])) * pair_grad,
                    tv_weighted_gradient_norm=abs(float(seam_cfg["weight_tv"])) * tv_grad,
                    gradient_norm=seam_grad, gradient_l1_sum=seam_gradient["l1"],
                    gradient_max_abs=seam_gradient["max_abs"],
                    gradient_finite=seam_gradient["finite"], reason=seam["reason"],
                )

                seam_summary["batches"] += 1
                seam_summary["samples"] += len(visibility_stats["mutual_pair_texels"])
                seam_summary["pair_valid_batches"] += int(mutual_count > 0)
                seam_summary["pair_valid_samples"] += sum(
                    count > 0 for count in visibility_stats["mutual_pair_texels"]
                )
                seam_summary["tv_valid_batches"] += int(tv_edge_count > 0)
                seam_summary["tv_valid_samples"] += sum(
                    count > 0 for count in visibility_stats["tv_effective_edges"]
                )
                for key in (
                    "candidate_pair_texels", "source_visible_pair_texels",
                    "partner_visible_pair_texels", "mutual_pair_texels",
                    "tv_effective_texels", "tv_effective_edges",
                ):
                    seam_summary[key] += sum(visibility_stats[key])
                seam_summary["pair_losses"].append(float(pair_loss.detach()))
                seam_summary["tv_losses"].append(float(tv_loss.detach()))
                seam_summary["pair_gradients"].append(pair_grad)
                seam_summary["tv_gradients"].append(tv_grad)
                seam_summary["weighted_pair_gradients"].append(
                    abs(float(seam_cfg["weight_pair"])) * pair_grad
                )
                seam_summary["weighted_tv_gradients"].append(
                    abs(float(seam_cfg["weight_tv"])) * tv_grad
                )
                seam_summary["per_batch_mutual_pair_texels"].append(
                    visibility_stats["mutual_pair_texels"]
                )
                seam_summary["uv_bake_assignment"].append(bake_assignment)
                for key in ("source_hits", "mutual_hits"):
                    if seam_summary[key] is None:
                        seam_summary[key] = visibility_stats[key].clone()
                    else:
                        seam_summary[key] += visibility_stats[key]
            else:
                report("seam", "SKIP", reason="disabled_by_config")

            trainer.optimizer.zero_grad(set_to_none=True)
            total = trainer.compute_loss(outputs)["loss"]
            total.backward()
            total_gradient = accumulated_gradient_statistics(parameters)
            total_grad = total_gradient["l2"]
            failed |= report(
                "trainer_total", "PASS" if torch.isfinite(total) and total_grad > 0
                and total_gradient["finite"] else "FAIL",
                loss=float(total.detach()), gradient_norm=total_grad,
                gradient_l1_sum=total_gradient["l1"],
                gradient_max_abs=total_gradient["max_abs"],
                gradient_finite=total_gradient["finite"],
                optimizer_step=False, checkpoint_written=False,
            )

        if "seam" in selected:
            pair_valid = trainer.seam_maps.get("pair_valid")
            atlas_mask = (pair_valid.detach().cpu() > 0) if pair_valid is not None else None
            atlas_pair_texels = int(atlas_mask.sum()) if atlas_mask is not None else 0
            atlas_total_texels = int(atlas_mask.numel()) if atlas_mask is not None else 0
            source_observed = (
                int(((seam_summary["source_hits"] > 0) & atlas_mask).sum())
                if seam_summary["source_hits"] is not None else 0
            )
            mutual_observed = (
                int(((seam_summary["mutual_hits"] > 0) & atlas_mask).sum())
                if seam_summary["mutual_hits"] is not None else 0
            )
            candidate_count = seam_summary["candidate_pair_texels"]
            pair_availability = (
                seam_summary["mutual_pair_texels"] / candidate_count if candidate_count else 0.0
            )
            mapped_mutual_ever = mutual_observed / atlas_pair_texels if atlas_pair_texels else 0.0
            pair_zero_batches = seam_summary["batches"] - seam_summary["pair_valid_batches"]
            if seam_summary["batches"] == 0 or seam_summary["pair_valid_batches"] == 0:
                summary_status = "FAIL"
                availability = "none"
            elif pair_zero_batches > 0:
                summary_status = "WARN"
                availability = "intermittent"
            else:
                summary_status = "PASS"
                availability = "all_checked_batches"

            def mapped_hit_distribution(hit_map):
                if hit_map is None or atlas_mask is None or atlas_pair_texels == 0:
                    return {"min": 0, "max": 0, "mean": 0.0, "q25": 0.0,
                            "median": 0.0, "q75": 0.0}
                values = hit_map.reshape(-1)[atlas_mask.reshape(-1)].float()
                quantiles = torch.quantile(
                    values, torch.tensor([0.25, 0.5, 0.75], dtype=values.dtype)
                )
                return {
                    "min": int(values.min()), "max": int(values.max()),
                    "mean": float(values.mean()), "q25": float(quantiles[0]),
                    "median": float(quantiles[1]), "q75": float(quantiles[2]),
                }

            failed |= report(
                "seam_multi_batch_summary", summary_status, pair_availability=availability,
                batches_checked=seam_summary["batches"], samples_checked=seam_summary["samples"],
                pair_valid_batches=seam_summary["pair_valid_batches"],
                pair_valid_samples=seam_summary["pair_valid_samples"],
                pair_zero_batches=pair_zero_batches,
                tv_valid_batches=seam_summary["tv_valid_batches"],
                tv_valid_samples=seam_summary["tv_valid_samples"],
                atlas_pair_texels=atlas_pair_texels, atlas_total_texels=atlas_total_texels,
                atlas_mapping_coverage=atlas_pair_texels / atlas_total_texels if atlas_total_texels else 0.0,
                source_observed_atlas_texels=source_observed,
                mutual_observed_atlas_texels=mutual_observed,
                never_mutual_atlas_texels=max(atlas_pair_texels - mutual_observed, 0),
                mapped_texels_ever_mutual_ratio=mapped_mutual_ever,
                mapped_source_hit_distribution=mapped_hit_distribution(seam_summary["source_hits"]),
                mapped_mutual_hit_distribution=mapped_hit_distribution(seam_summary["mutual_hits"]),
                candidate_pair_texels=candidate_count,
                source_visible_pair_texels=seam_summary["source_visible_pair_texels"],
                partner_visible_pair_texels=seam_summary["partner_visible_pair_texels"],
                mutual_pair_texels=seam_summary["mutual_pair_texels"],
                overall_mutual_pair_ratio=pair_availability,
                tv_effective_texels=seam_summary["tv_effective_texels"],
                tv_effective_edges=seam_summary["tv_effective_edges"],
                pair_losses=seam_summary["pair_losses"], tv_losses=seam_summary["tv_losses"],
                pair_gradient_norms=seam_summary["pair_gradients"],
                tv_gradient_norms=seam_summary["tv_gradients"],
                weighted_pair_gradient_norms=seam_summary["weighted_pair_gradients"],
                weighted_tv_gradient_norms=seam_summary["weighted_tv_gradients"],
                per_batch_mutual_pair_texels=seam_summary["per_batch_mutual_pair_texels"],
                uv_bake_assignment=seam_summary["uv_bake_assignment"],
                manifest_sha256=manifest["sha256"], sampler_mode=sampler_mode,
                interpretation=(
                    "pair_zero_batches_have_no_mutual_visibility"
                    if pair_zero_batches > 0 else "all_checked_batches_have_mutual_visibility"
                ),
            )

        written_checkpoints = list(Path(workspace).glob("*.pth"))
        failed |= report(
            "no_checkpoint_write", "PASS" if not written_checkpoints else "FAIL",
            paths=[str(path) for path in written_checkpoints],
        )

    if REPORT_HANDLE is not None:
        REPORT_HANDLE.close()
        REPORT_HANDLE = None
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
