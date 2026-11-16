import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TrainerWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = (ROOT / "joint_trainer.py").read_text()
        cls.train = (ROOT / "train.py").read_text()
        cls.smoke = (ROOT / "tools/fresco_smoke_test.py").read_text()
        cls.seam_compare = (ROOT / "tools/compare_seam_smoke.py").read_text()
        ast.parse(cls.trainer)
        ast.parse(cls.train)
        ast.parse(cls.smoke)
        ast.parse(cls.seam_compare)

    def test_cross_view_sampler_is_used_by_dataloader(self):
        self.assertIn("batch_sampler=pair_sampler", self.train)
        self.assertIn("paired_sampler_required(config)", self.train)

    def test_frequency_and_uv_losses_are_added_once(self):
        self.assertEqual(self.trainer.count('loss = loss + lf_weight * lf_stats["loss"]'), 1)
        self.assertEqual(self.trainer.count('loss = loss + hf_weight * hf_stats["loss"]'), 1)
        self.assertEqual(self.trainer.count('loss = loss + uv_weight * uv_stats["loss"]'), 1)
        self.assertEqual(self.trainer.count('loss = loss + seam_stats["loss"]'), 1)

    def test_epoch_checkpoint_uses_next_epoch(self):
        increment = self.trainer.index("self.current_epoch += 1")
        checkpoint = self.trainer.index("Epoch boundary checkpoint saved")
        self.assertLess(increment, checkpoint)

    def test_evaluation_does_not_restore_adam_or_scheduler(self):
        utils = (ROOT / "utils.py").read_text()
        self.assertIn("restore_training_state = not self.is_val", self.trainer)
        self.assertIn("restore_training_state=restore_training_state", self.trainer)
        self.assertIn("if restore_training_state:", utils)
        self.assertIn("Evaluation restore: skipped optimizer and scheduler state", utils)

    def test_active_uv_path_is_cross_view(self):
        self.assertIn("self._prepare_cross_view_uv(outputs)", self.trainer)
        self.assertNotIn("self._maybe_uv_bake(items, outputs)", self.trainer)

    def test_flame_restore_preserves_optimizer_tensor_identity(self):
        self.assertIn("destination.copy_(source)", self.trainer)
        load_method = self.trainer.split("def load_all_flame_params", 1)[1].split("def load_flame_params", 1)[0]
        self.assertNotIn("self.all_flame_params = {k: torch.from_numpy", load_method)

    def test_disabled_seam_does_not_initialize_maps(self):
        self.assertIn('if not self.config.get("exp.lf_uv.seam.enable", False):', self.trainer)
        self.assertIn("seam_maps_v2_", self.trainer)

    def test_smoke_manifest_replay_is_independent_of_training_sampler(self):
        self.assertIn('choices=("auto", "paired", "sequential")', self.smoke)
        self.assertIn("ManifestBatchSampler(manifest_batches)", self.smoke)
        self.assertIn("set_reproducible_seed(args.seed)", self.smoke)
        self.assertNotIn("optimizer.step(", self.smoke)

    def test_seam_comparison_checks_raw_components_and_visibility(self):
        self.assertIn('"mutual_pair_texels"', self.seam_compare)
        self.assertIn('"gradient_norm", "weighted_gradient_norm"', self.seam_compare)

    def test_smoke_rejects_missing_checkpoint_before_trainer_construction(self):
        checkpoint_check = self.smoke.index("checkpoint_path.is_file()")
        trainer_construction = self.smoke.index("trainer = JointTrainer(")
        self.assertLess(checkpoint_check, trainer_construction)
        self.assertIn('initialization_mode = "random_initialization"', self.smoke)

    def test_smoke_reports_true_l2_and_weighted_seam_gradients(self):
        self.assertIn('"l2": math.sqrt(squared_sum)', self.smoke)
        self.assertIn("weighted_gradient_norm", self.smoke)
        self.assertIn("gradient_finite", self.smoke)

if __name__ == "__main__":
    unittest.main()
