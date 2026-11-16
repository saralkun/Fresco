"""Real NeRSemble Dataset/DataLoader checks for FReSCO cross-view supervision."""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.nersemble import NeRSembleData
from exp.config_utils import normalize_dotted_config
from exp.cross_view_sampler import SameFramePairBatchSampler


def report(name, status, **details):
    print(json.dumps({"test": name, "status": status, **details}, sort_keys=True))
    return status == "FAIL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-sizes", default="2,16")
    parser.add_argument("--max-batches", type=int, default=3)
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    overrides = {"data.num_workers": 0}
    config.update(overrides)
    normalize_dotted_config(config, overrides)
    config["bald"] = Path(args.config).parent.name == "218"

    failed = False
    dataset = NeRSembleData(config)
    required = {
        "timestep_index", "camera_index", "camera_id", "flame_param_path",
        "file_path", "transform_matrix",
    }
    missing = [index for index, frame in enumerate(dataset.framelist) if required - set(frame)]
    failed |= report(
        "framelist_schema", "PASS" if not missing else "FAIL",
        samples=len(dataset), malformed_indices=missing[:10],
    )

    groups = defaultdict(list)
    for index, frame in enumerate(dataset.framelist):
        groups[int(frame["timestep_index"])].append((index, frame))
    geometry_mismatch = []
    duplicate_cameras = []
    duplicate_camera_indices = []
    for timestep, entries in groups.items():
        paths = {str(frame["flame_param_path"]) for _, frame in entries}
        cameras = [str(frame["camera_id"]) for _, frame in entries]
        camera_indices = [int(frame["camera_index"]) for _, frame in entries]
        if len(paths) != 1:
            geometry_mismatch.append(timestep)
        if len(cameras) != len(set(cameras)):
            duplicate_cameras.append(timestep)
        if len(camera_indices) != len(set(camera_indices)):
            duplicate_camera_indices.append(timestep)
    failed |= report(
        "timestep_geometry_metadata",
        "PASS" if not geometry_mismatch and not duplicate_cameras and not duplicate_camera_indices else "FAIL",
        timesteps=len(groups), geometry_mismatch=geometry_mismatch[:10],
        duplicate_camera_timesteps=duplicate_cameras[:10],
        duplicate_camera_index_timesteps=duplicate_camera_indices[:10],
        views_min=min(len(entries) for entries in groups.values()),
        views_max=max(len(entries) for entries in groups.values()),
    )

    flame_files = {
        os.path.join(dataset.basedir, str(frame["flame_param_path"]))
        for frame in dataset.framelist
    }
    missing_flame_files = [path for path in flame_files if not os.path.exists(path)]
    failed |= report(
        "flame_parameter_files", "PASS" if not missing_flame_files else "FAIL",
        unique_files=len(flame_files), missing_count=len(missing_flame_files),
        missing_preview=missing_flame_files[:5],
    )

    sample = dataset[0]
    sample_ok = (
        sample["img"].shape == (config["data.img_h"], config["data.img_w"], 3)
        and sample["obj_mask"].shape == sample["img"].shape[:2]
        and sample["intr"].shape == (3, 3)
        and sample["w2c"].shape == (4, 4)
        and sample["img"].min() >= 0.0 and sample["img"].max() <= 1.0
    )
    failed |= report(
        "dataset_getitem", "PASS" if sample_ok else "FAIL",
        keys=sorted(sample), image_shape=list(sample["img"].shape),
        image_min=float(sample["img"].min()), image_max=float(sample["img"].max()),
        frame_idx=int(sample["frame_idx"]), camera_id=str(sample["cam"]),
    )

    for batch_size in [int(value) for value in args.batch_sizes.split(",")]:
        sampler = SameFramePairBatchSampler(dataset, batch_size, shuffle=False, drop_last=False)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        checked = 0
        pairing_ok = True
        collate_ok = True
        for batch in loader:
            if checked >= args.max_batches:
                break
            frames = torch.as_tensor(batch["frame_idx"]).tolist()
            cameras = [str(value) for value in batch["cam"]]
            collate_ok &= batch["img"].ndim == 4 and batch["img"].shape[-1] == 3
            collate_ok &= len(frames) == len(cameras) and len(frames) % 2 == 0
            for offset in range(0, len(frames), 2):
                pairing_ok &= frames[offset] == frames[offset + 1]
                pairing_ok &= cameras[offset] != cameras[offset + 1]
            checked += 1
        failed |= report(
            f"dataloader_batch_{batch_size}",
            "PASS" if pairing_ok and collate_ok and checked > 0 else "FAIL",
            checked_batches=checked, loader_batches=len(loader),
            source_samples=sampler.num_source_samples,
            emitted_samples=sampler.num_emitted_samples,
            sample_delta=sampler.epoch_sample_delta,
            excluded_single_view=sampler.num_excluded_single_view_samples,
            pairing_ok=pairing_ok, collate_ok=collate_ok,
        )

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
