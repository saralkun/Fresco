"""Batch sampler that guarantees same-timestep, different-camera pairs."""

import random
from collections import defaultdict

from torch.utils.data import Sampler


class SameFramePairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Cross-view UV supervision requires an even batch_size >= 2")
        if not hasattr(dataset, "framelist"):
            raise RuntimeError(
                "Dataset must expose framelist entries with timestep_index and camera_id "
                "to construct valid cross-view pairs"
            )
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        groups = defaultdict(list)
        flame_paths = defaultdict(set)
        camera_indices_by_id = defaultdict(set)
        timestep_camera_indices = defaultdict(list)
        for index, frame in enumerate(dataset.framelist):
            missing = {"timestep_index", "camera_id"} - set(frame)
            if missing:
                raise RuntimeError(f"framelist[{index}] is missing required fields: {sorted(missing)}")
            groups[int(frame["timestep_index"])].append((str(frame["camera_id"]), index))
            if frame.get("camera_index") is not None:
                camera_indices_by_id[str(frame["camera_id"])].add(int(frame["camera_index"]))
                timestep_camera_indices[int(frame["timestep_index"])].append(int(frame["camera_index"]))
            if frame.get("flame_param_path") is not None:
                flame_paths[int(frame["timestep_index"])].add(str(frame["flame_param_path"]))
        inconsistent = {frame: paths for frame, paths in flame_paths.items() if len(paths) > 1}
        if inconsistent:
            preview = list(inconsistent.items())[:3]
            raise RuntimeError(f"A timestep references multiple FLAME parameter files: {preview}")
        duplicate_cameras = {
            frame: [cam for cam, _ in items]
            for frame, items in groups.items()
            if len(items) != len({cam for cam, _ in items})
        }
        if duplicate_cameras:
            preview = list(duplicate_cameras.items())[:3]
            raise RuntimeError(f"A timestep contains duplicate camera_id entries: {preview}")
        inconsistent_camera_indices = {
            camera: indices for camera, indices in camera_indices_by_id.items() if len(indices) > 1
        }
        duplicate_camera_indices = {
            frame: indices for frame, indices in timestep_camera_indices.items()
            if len(indices) != len(set(indices))
        }
        if inconsistent_camera_indices or duplicate_camera_indices:
            raise RuntimeError(
                "camera_id/camera_index metadata is inconsistent: "
                f"id_mapping={list(inconsistent_camera_indices.items())[:3]} "
                f"duplicate_indices={list(duplicate_camera_indices.items())[:3]}"
            )

        self.groups = [items for items in groups.values() if len({cam for cam, _ in items}) >= 2]
        self.num_dataset_samples = len(dataset)
        self.num_excluded_single_view_samples = self.num_dataset_samples - sum(len(items) for items in self.groups)
        if not self.groups:
            raise RuntimeError("No timestep has at least two distinct camera views")
        self.num_source_samples = sum(len(items) for items in self.groups)
        self.num_odd_groups = sum(len(items) % 2 for items in self.groups)
        # Alternating odd groups omit or duplicate one view. This preserves the
        # original epoch size exactly when the number of odd groups is even and
        # differs by at most one sample otherwise, instead of adding one sample
        # for every odd-camera timestep.
        self.num_duplicated_odd_groups = self.num_odd_groups // 2
        self.num_pairs = (
            sum(len(items) // 2 for items in self.groups) + self.num_duplicated_odd_groups
        )
        self.num_emitted_samples = self.num_pairs * 2
        self.epoch_sample_delta = self.num_emitted_samples - self.num_source_samples
        self._epoch = 0

    @staticmethod
    def _pair_group(entries, duplicate_tail):
        """Pair distinct cameras, optionally duplicating an odd tail."""
        pairs = []
        remaining = list(entries)
        while len(remaining) >= 2:
            first = remaining.pop()
            partner_index = next(
                (i for i, item in enumerate(remaining) if item[0] != first[0]), None
            )
            if partner_index is None:
                break
            second = remaining.pop(partner_index)
            pairs.append((first[1], second[1]))
        if remaining and duplicate_tail:
            first = remaining[0]
            second = next(item for item in entries if item[0] != first[0])
            pairs.append((first[1], second[1]))
        return pairs

    def __iter__(self):
        groups = list(enumerate(self.groups))
        if self.shuffle:
            random.shuffle(groups)
        odd_group_ids = [group_id for group_id, entries in groups if len(entries) % 2]
        if odd_group_ids:
            # Rotate the omit/duplicate assignment across epochs so no timestep
            # or camera is systematically underrepresented.
            shift = self._epoch % len(odd_group_ids)
            odd_group_ids = odd_group_ids[shift:] + odd_group_ids[:shift]
        duplicate_ids = set(odd_group_ids[: self.num_duplicated_odd_groups])
        pairs = []
        for group_id, entries in groups:
            entries = list(entries)
            if self.shuffle:
                random.shuffle(entries)
            elif entries:
                shift = self._epoch % len(entries)
                entries = entries[shift:] + entries[:shift]
            pairs.extend(self._pair_group(entries, group_id in duplicate_ids))
        if self.shuffle:
            random.shuffle(pairs)
        self._epoch += 1
        batch = []
        for first, second in pairs:
            batch.extend((first, second))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        pairs_per_batch = self.batch_size // 2
        full, remainder = divmod(self.num_pairs, pairs_per_batch)
        return full + (bool(remainder) and not self.drop_last)
