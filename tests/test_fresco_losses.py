import unittest

import torch

from exp.config_utils import normalize_dotted_config
from exp.cross_view_sampler import SameFramePairBatchSampler
from exp.cross_view_uv_loss import bake_attribute_to_uv, compute_cross_view_uv_loss
from exp.hfimg_loss import compute_hfimg_loss, difference_of_gaussians, gradient_difference_loss
from exp.lfimg_loss import compute_lfimg_loss
from exp.uv_seam_loss import compute_seam_loss
from geom.seam_pairs import build_seam_pair_maps


class ImageFrequencyLossTests(unittest.TestCase):
    def test_hf_rejects_missing_sigma_instead_of_using_a_default(self):
        pred = torch.zeros(1, 3, 4, 4)
        gt = torch.zeros_like(pred)
        cfg = {"kernel1": 3, "sigma1": None, "kernel2": 5, "sigma2": 1.2,
               "lambda_h1": 1.0, "lambda_h2": 1.0}
        with self.assertRaisesRegex(ValueError, "sigma1"):
            compute_hfimg_loss(pred, gt, None, cfg)

    def test_lf_matches_direct_l1_and_backpropagates(self):
        pred = torch.rand(2, 16, 16, 3, requires_grad=True)
        gt = torch.rand_like(pred)
        stats = compute_lfimg_loss(pred, gt, cfg={"kernel": 5, "sigma": 1.2})
        self.assertTrue(torch.isfinite(stats["loss"]))
        stats["loss"].backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(pred.grad.abs().sum().item(), 0)

    def test_lf_does_not_cut_gradient_outside_display_range(self):
        pred = torch.full((1, 3, 8, 8), 1.5, requires_grad=True)
        stats = compute_lfimg_loss(pred, torch.zeros_like(pred), cfg={"kernel": 3, "sigma": 0.8})
        stats["loss"].backward()
        self.assertGreater(pred.grad.abs().sum().item(), 0)

    def test_hf_is_dog_edge_plus_gdl(self):
        pred = torch.rand(1, 3, 12, 12, requires_grad=True)
        gt = torch.rand_like(pred)
        cfg = {"kernel1": 3, "sigma1": 0.8, "kernel2": 5, "sigma2": 1.2,
               "lambda_h1": 0.7, "lambda_h2": 0.3}
        stats = compute_hfimg_loss(pred, gt, cfg=cfg)
        expected = cfg["lambda_h1"] * stats["edge_hf"] + cfg["lambda_h2"] * stats["gdl"]
        self.assertTrue(torch.allclose(stats["loss"].detach(), expected))
        self.assertTrue(torch.isfinite(stats["loss"]))
        self.assertEqual(difference_of_gaussians(pred, 3, 0.8, 5, 1.2).shape, pred.shape)
        self.assertTrue(torch.isfinite(gradient_difference_loss(pred, gt)))
        stats["loss"].backward()
        self.assertGreater(pred.grad.abs().sum().item(), 0)

    def test_gdl_uses_combined_2d_gradient_magnitude(self):
        single_channel = torch.tensor([[[[0.0, 1.0], [1.0, 1.0]]]])
        pred = single_channel.expand(-1, 3, -1, -1).clone()
        gt = torch.zeros_like(pred)
        dx = torch.nn.functional.pad(pred[..., 1:] - pred[..., :-1], (0, 1, 0, 0))
        dy = torch.nn.functional.pad(pred[..., 1:, :] - pred[..., :-1, :], (0, 0, 0, 1))
        expected = (
            torch.sqrt(dx.square() + dy.square() + 1e-12)
            - torch.sqrt(torch.full_like(dx, 1e-12))
        ).abs().mean()
        self.assertTrue(torch.allclose(gradient_difference_loss(pred, gt), expected))

    def test_gdl_flat_image_has_finite_zero_gradient(self):
        pred = torch.zeros(1, 3, 4, 4, requires_grad=True)
        loss = gradient_difference_loss(pred, torch.zeros_like(pred))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(pred.grad).all())


class CrossViewUvTests(unittest.TestCase):
    def _inputs(self):
        visibility = torch.ones(2, 1, 4, 4)
        confidence = torch.ones(2, 1, 4, 4)
        return visibility, confidence, torch.tensor([7, 7]), ["cam_a", "cam_b"]

    def test_equal_content_is_zero(self):
        texture = torch.ones(2, 3, 4, 4, requires_grad=True)
        result = compute_cross_view_uv_loss(texture, *self._inputs())
        self.assertAlmostEqual(result["loss"].item(), 0.0, places=7)

    def test_difference_increases_and_target_is_stop_gradient(self):
        texture = torch.zeros(2, 3, 4, 4)
        texture[1] = 1.0
        texture.requires_grad_()
        result = compute_cross_view_uv_loss(texture, *self._inputs())
        self.assertGreater(result["loss"].item(), 0.0)
        result["loss"].backward()
        self.assertGreater(texture.grad[0].abs().sum().item(), 0.0)
        self.assertEqual(texture.grad[1].abs().sum().item(), 0.0)

    def test_no_mutual_visibility_is_safe(self):
        texture = torch.rand(2, 3, 4, 4, requires_grad=True)
        visibility, confidence, frames, cameras = self._inputs()
        visibility[1].zero_()
        result = compute_cross_view_uv_loss(texture, visibility, confidence, frames, cameras)
        self.assertTrue(result["skipped"])
        self.assertEqual(result["reason"], "no_mutual_visibility")
        result["loss"].backward()
        self.assertEqual(texture.grad.abs().sum().item(), 0.0)

    def test_zero_view_confidence_is_reported(self):
        texture = torch.rand(2, 3, 4, 4, requires_grad=True)
        visibility, confidence, frames, cameras = self._inputs()
        confidence.zero_()
        result = compute_cross_view_uv_loss(texture, visibility, confidence, frames, cameras)
        self.assertTrue(result["skipped"])
        self.assertEqual(result["reason"], "zero_view_confidence")

    def test_uv_splat_is_aligned_and_differentiable(self):
        attribute = torch.zeros(1, 2, 2, 1, requires_grad=True)
        attribute.data[0, 0, 0, 0] = 0.75
        face_id = torch.full((1, 2, 2), -1, dtype=torch.long)
        face_id[0, 0, 0] = 0
        face_bw = torch.zeros(1, 2, 2, 2)
        face_bw[0, 0, 0, 0] = 1.0
        vertex_uvs = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
        faces_uvs = torch.tensor([[0, 1, 2]], dtype=torch.long)
        baked, mask = bake_attribute_to_uv(
            attribute, face_id, face_bw, vertex_uvs, faces_uvs, face_id >= 0, (4, 4)
        )
        self.assertAlmostEqual(baked[0, 0, 1, 1].item(), 0.75)
        self.assertEqual(mask.sum().item(), 1.0)
        baked.sum().backward()
        self.assertGreater(attribute.grad[0, 0, 0, 0].item(), 0.0)


class CrossViewSamplerTests(unittest.TestCase):
    class Dataset:
        def __init__(self, views_per_frame):
            self.framelist = [
                {"timestep_index": frame, "camera_id": f"cam_{camera}",
                 "camera_index": camera,
                 "flame_param_path": f"flame/{frame}.npz"}
                for frame in range(3) for camera in range(views_per_frame)
            ]

        def __len__(self):
            return len(self.framelist)

    def test_even_camera_count_preserves_epoch_samples(self):
        dataset = self.Dataset(4)
        sampler = SameFramePairBatchSampler(dataset, batch_size=4, shuffle=False)
        emitted = [index for batch in sampler for index in batch]
        self.assertEqual(sampler.num_source_samples, 12)
        self.assertEqual(sampler.num_emitted_samples, 12)
        self.assertCountEqual(emitted, range(12))

    def test_odd_camera_count_preserves_epoch_scale_and_rotates_coverage(self):
        dataset = self.Dataset(3)
        sampler = SameFramePairBatchSampler(dataset, batch_size=4, shuffle=False)
        emitted_epoch_1 = [index for batch in sampler for index in batch]
        emitted_epoch_2 = [index for batch in sampler for index in batch]
        self.assertEqual(sampler.num_source_samples, 9)
        self.assertEqual(sampler.num_emitted_samples, 8)
        self.assertEqual(sampler.epoch_sample_delta, -1)
        self.assertTrue(set(range(9)).issubset(set(emitted_epoch_1) | set(emitted_epoch_2)))
        for batch in sampler:
            for offset in range(0, len(batch), 2):
                left, right = dataset.framelist[batch[offset]], dataset.framelist[batch[offset + 1]]
                self.assertEqual(left["timestep_index"], right["timestep_index"])
                self.assertNotEqual(left["camera_id"], right["camera_id"])

    def test_fifteen_views_does_not_expand_epoch(self):
        dataset = self.Dataset(15)
        sampler = SameFramePairBatchSampler(dataset, batch_size=16, shuffle=False)
        self.assertEqual(sampler.num_source_samples, 45)
        self.assertEqual(sampler.num_emitted_samples, 44)
        self.assertEqual(len(sampler), 3)

    def test_conflicting_flame_paths_are_rejected(self):
        dataset = self.Dataset(2)
        dataset.framelist[1]["flame_param_path"] = "flame/wrong.npz"
        with self.assertRaisesRegex(RuntimeError, "multiple FLAME"):
            SameFramePairBatchSampler(dataset, batch_size=2)


class SeamAndConfigTests(unittest.TestCase):
    def test_seam_map_matches_reversed_triangle_endpoints(self):
        faces = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)
        faces_uvs = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
        vertex_uvs = torch.tensor([
            [0.1, 0.2], [0.1, 0.8], [0.3, 0.5],
            [0.8, 0.8], [0.8, 0.2], [0.6, 0.5],
        ])
        maps = build_seam_pair_maps(faces, vertex_uvs, faces_uvs, (8, 8))
        source_y, source_x = int(0.2 * 7), int(0.1 * 7)
        target_y, target_x = int(0.2 * 7), int(0.8 * 7)
        self.assertEqual(maps["pair_valid"][0, 0, source_y, source_x].item(), 1.0)
        self.assertAlmostEqual(
            maps["pair_u"][0, 0, source_y, source_x].item(),
            2.0 * target_x / 7.0 - 1.0,
        )
        self.assertAlmostEqual(
            maps["pair_v"][0, 0, source_y, source_x].item(),
            2.0 * target_y / 7.0 - 1.0,
        )

    def test_seam_contract_and_gradient(self):
        texture = torch.rand(1, 3, 4, 4, requires_grad=True)
        xs = torch.linspace(-1, 1, 4).view(1, 1, 1, 4).expand(1, 1, 4, 4)
        ys = torch.linspace(-1, 1, 4).view(1, 1, 4, 1).expand(1, 1, 4, 4)
        valid = torch.ones(1, 1, 4, 4)
        stats = compute_seam_loss(texture, {"weight_pair": 0.02, "weight_tv": 0.004},
                                  uv_mask=valid, pair_u=-xs, pair_v=ys, pair_valid=valid)
        self.assertFalse(stats["skipped"])
        self.assertIn("pair", stats)
        self.assertIn("tv", stats)
        stats["loss"].backward()
        self.assertGreater(texture.grad.abs().sum().item(), 0)

    def test_seam_pair_requires_both_sides_visible(self):
        texture = torch.rand(1, 3, 4, 4, requires_grad=True)
        xs = torch.linspace(-1, 1, 4).view(1, 1, 1, 4).expand(1, 1, 4, 4)
        ys = torch.linspace(-1, 1, 4).view(1, 1, 4, 1).expand(1, 1, 4, 4)
        source_only = torch.zeros(1, 1, 4, 4)
        source_only[..., :2] = 1.0
        valid = torch.ones_like(source_only)
        stats = compute_seam_loss(
            texture, {"weight_pair": 1.0, "weight_tv": 0.0}, uv_mask=source_only,
            pair_u=-xs, pair_v=ys, pair_valid=valid,
        )
        self.assertEqual(stats["pair_cov"], 0.0)
        self.assertEqual(stats["pair"].item(), 0.0)

    def test_flat_override_wins_and_syncs_nested(self):
        config = {"exp": {"lf_img": {"lambda": 0.25}}, "exp.lf_img.lambda": 0.1}
        with self.assertWarns(RuntimeWarning):
            normalize_dotted_config(config)
        self.assertEqual(config["exp.lf_img.lambda"], 0.1)
        self.assertEqual(config["exp"]["lf_img"]["lambda"], 0.1)


if __name__ == "__main__":
    unittest.main()
