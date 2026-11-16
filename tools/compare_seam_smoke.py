#!/usr/bin/env python3
"""Compare Seam diagnostics from two smoke runs using the same manifest."""

import argparse
import json
import math
from pathlib import Path


def load_reports(path):
    reports = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                reports.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {error}") from error
    return reports


def one_report(reports, test):
    matches = [report for report in reports if report.get("test") == test]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {test!r} report, found {len(matches)}")
    return matches[0]


def batch_reports(reports, test):
    result = {}
    for report in reports:
        if report.get("test") != test:
            continue
        batch_index = report.get("batch_index")
        if batch_index in result:
            raise ValueError(f"Duplicate {test!r} report for batch {batch_index}")
        result[batch_index] = report
    return result


def close(left, right, rtol, atol):
    return math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args()

    left_reports = load_reports(args.left)
    right_reports = load_reports(args.right)
    left_manifest = one_report(left_reports, "batch_manifest")
    right_manifest = one_report(right_reports, "batch_manifest")
    failed = False
    manifest_match = left_manifest["manifest_sha256"] == right_manifest["manifest_sha256"]
    print(json.dumps({
        "comparison": "manifest", "status": "PASS" if manifest_match else "FAIL",
        "left_sha256": left_manifest["manifest_sha256"],
        "right_sha256": right_manifest["manifest_sha256"],
    }, sort_keys=True))
    failed |= not manifest_match
    left_run = one_report(left_reports, "run_configuration")
    right_run = one_report(right_reports, "run_configuration")
    left_checkpoint = one_report(left_reports, "checkpoint_input")
    right_checkpoint = one_report(right_reports, "checkpoint_input")
    left_resume = one_report(left_reports, "resume_state")
    right_resume = one_report(right_reports, "resume_state")
    state_match = (
        left_run.get("checkpoint") == right_run.get("checkpoint")
        and left_run.get("seed") == right_run.get("seed")
        and left_run.get("sampler_mode") == right_run.get("sampler_mode")
        and left_run.get("initialization_mode") == right_run.get("initialization_mode")
    )
    initialization_mode = left_run.get("initialization_mode")
    if initialization_mode == "checkpoint":
        restoration_ok = (
            left_checkpoint.get("status") == "PASS"
            and right_checkpoint.get("status") == "PASS"
            and left_resume.get("restore_verified") is True
            and right_resume.get("restore_verified") is True
        )
    else:
        restoration_ok = (
            initialization_mode == "random_initialization"
            and left_checkpoint.get("initialization_mode") == "random_initialization"
            and right_checkpoint.get("initialization_mode") == "random_initialization"
            and left_run.get("seed") == right_run.get("seed")
        )
    state_match = state_match and restoration_ok
    print(json.dumps({
        "comparison": "model_and_sampling_state",
        "status": "PASS" if state_match else "FAIL",
        "checkpoint_left": left_run.get("checkpoint"),
        "checkpoint_right": right_run.get("checkpoint"),
        "seed_left": left_run.get("seed"), "seed_right": right_run.get("seed"),
        "sampler_mode_left": left_run.get("sampler_mode"),
        "sampler_mode_right": right_run.get("sampler_mode"),
        "initialization_mode": initialization_mode,
        "checkpoint_restore_verified_left": left_resume.get("restore_verified"),
        "checkpoint_restore_verified_right": right_resume.get("restore_verified"),
    }, sort_keys=True))
    failed |= not state_match

    left_mapping = one_report(left_reports, "seam_atlas_mapping")
    right_mapping = one_report(right_reports, "seam_atlas_mapping")
    mapping_match = left_mapping == right_mapping
    print(json.dumps({
        "comparison": "seam_atlas_mapping", "status": "PASS" if mapping_match else "FAIL",
        "atlas_pair_texels_left": left_mapping.get("atlas_pair_texels"),
        "atlas_pair_texels_right": right_mapping.get("atlas_pair_texels"),
        "reciprocal_pairs_left": left_mapping.get("reciprocal_pairs"),
        "reciprocal_pairs_right": right_mapping.get("reciprocal_pairs"),
    }, sort_keys=True))
    failed |= not mapping_match

    tests = ("seam_pair", "seam_tv", "seam", "uv_bake_assignment")
    indexed = {
        test: (batch_reports(left_reports, test), batch_reports(right_reports, test))
        for test in tests
    }
    all_batches = sorted(set(indexed["seam_pair"][0]) | set(indexed["seam_pair"][1]))
    for batch_index in all_batches:
        differences = []
        for test, (left_by_batch, right_by_batch) in indexed.items():
            left = left_by_batch.get(batch_index)
            right = right_by_batch.get(batch_index)
            if left is None or right is None:
                differences.append(f"{test}:missing_report")
                continue
            for key in ("dataset_indices", "timesteps"):
                if key in left or key in right:
                    if left.get(key) != right.get(key):
                        differences.append(f"{test}:{key}")
            if test == "seam_pair":
                for key in (
                    "camera_ids", "candidate_pair_texels", "source_visible_pair_texels",
                    "partner_visible_pair_texels", "mutual_pair_texels", "reason",
                    "reciprocal_mutual_pair_texels", "nonreciprocal_mutual_pair_texels",
                    "duplicate_target_mutual_pair_texels",
                    "source_boundary_mutual_pair_texels", "partner_boundary_mutual_pair_texels",
                ):
                    if left.get(key) != right.get(key):
                        differences.append(f"{test}:{key}")
            if test == "uv_bake_assignment":
                for key in (
                    "samples", "geometry_source_visible_pair_texels",
                    "geometry_partner_visible_pair_texels", "geometry_mutual_pair_texels",
                    "final_source_visible_pair_texels", "final_partner_visible_pair_texels",
                    "final_mutual_pair_texels",
                ):
                    if left.get(key) != right.get(key):
                        differences.append(f"{test}:{key}")
            if test in ("seam_pair", "seam_tv"):
                for key in ("loss", "gradient_norm", "weighted_gradient_norm", "gradient_max_abs"):
                    if not close(left[key], right[key], args.rtol, args.atol):
                        differences.append(
                            f"{test}:{key}({left[key]:.9g}!={right[key]:.9g})"
                        )
            if test == "seam":
                for key in (
                    "loss", "pair", "tv", "pair_gradient_norm", "tv_gradient_norm",
                    "pair_weighted_gradient_norm", "tv_weighted_gradient_norm",
                ):
                    if not close(left[key], right[key], args.rtol, args.atol):
                        differences.append(
                            f"{test}:{key}({left[key]:.9g}!={right[key]:.9g})"
                        )
        status = "PASS" if not differences else "FAIL"
        failed |= bool(differences)
        left_pair = indexed["seam_pair"][0].get(batch_index, {})
        right_pair = indexed["seam_pair"][1].get(batch_index, {})
        print(json.dumps({
            "comparison": "batch", "batch_index": batch_index, "status": status,
            "dataset_indices": left_pair.get("dataset_indices"),
            "mutual_pair_texels_left": left_pair.get("mutual_pair_texels"),
            "mutual_pair_texels_right": right_pair.get("mutual_pair_texels"),
            "differences": differences,
        }, sort_keys=True))

    left_summary = one_report(left_reports, "seam_multi_batch_summary")
    right_summary = one_report(right_reports, "seam_multi_batch_summary")
    summary_differences = []
    for key in (
        "batches_checked", "samples_checked", "pair_valid_batches", "pair_valid_samples",
        "pair_zero_batches", "tv_valid_batches", "tv_valid_samples",
        "candidate_pair_texels", "source_visible_pair_texels",
        "partner_visible_pair_texels", "mutual_pair_texels",
        "per_batch_mutual_pair_texels",
    ):
        if left_summary.get(key) != right_summary.get(key):
            summary_differences.append(key)
    for key in ("overall_mutual_pair_ratio", "mapped_texels_ever_mutual_ratio"):
        if not close(left_summary.get(key, 0.0), right_summary.get(key, 0.0), args.rtol, args.atol):
            summary_differences.append(key)
    summary_match = not summary_differences
    failed |= not summary_match
    print(json.dumps({
        "comparison": "seam_multi_batch_summary",
        "status": "PASS" if summary_match else "FAIL",
        "pair_valid_batches_left": left_summary.get("pair_valid_batches"),
        "pair_valid_batches_right": right_summary.get("pair_valid_batches"),
        "overall_mutual_pair_ratio_left": left_summary.get("overall_mutual_pair_ratio"),
        "overall_mutual_pair_ratio_right": right_summary.get("overall_mutual_pair_ratio"),
        "differences": summary_differences,
    }, sort_keys=True))

    print(json.dumps({
        "comparison": "summary", "status": "FAIL" if failed else "PASS",
        "batches_compared": len(all_batches), "rtol": args.rtol, "atol": args.atol,
        "criterion": "same manifest, identities, visibility counts, raw Seam values, and gradients",
    }, sort_keys=True))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
