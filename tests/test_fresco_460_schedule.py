import ast
import unittest
from pathlib import Path

from exp.fresco_stage_schedule import (
    SUBJECT460_100EP,
    apply_subject460_stage,
    paired_sampler_required,
    stage_for_epoch,
)


ROOT = Path(__file__).resolve().parents[1]


def base_config():
    return {
        "exp.fresco_schedule.name": "subject460_100ep",
        "exp.hf_img.kernel1": 3,
        "exp.hf_img.sigma1": 0.8,
        "exp.hf_img.kernel2": 5,
        "exp.hf_img.sigma2": 1.2,
        "exp.hf_img.lambda_h1": 1.0,
        "exp.hf_img.lambda_h2": 1.0,
        "exp.lf_uv.seam.weight_tv": 0.004,
    }


class Subject460ScheduleTests(unittest.TestCase):
    def test_exact_stage_table(self):
        actual = [(s.start, s.end, s.lf, s.uv, s.seam, s.hf) for s in SUBJECT460_100EP]
        self.assertEqual(actual, [
            (0, 10, .25, 0, 0, 0), (10, 30, .15, .01, 0, 0),
            (30, 48, .10, .02, 0, 0), (48, 70, .05, .03, .05, 0),
            (70, 90, .02, .03, .02, .05), (90, 100, .01, .02, .01, .05),
        ])

    def test_all_boundaries(self):
        expected = {9: 0, 10: 10, 29: 10, 30: 30, 47: 30, 48: 48,
                    69: 48, 70: 70, 89: 70, 90: 90, 99: 90}
        for epoch, start in expected.items():
            self.assertEqual(stage_for_epoch(epoch).start, start)
        with self.assertRaises(ValueError):
            stage_for_epoch(100)

    def test_sampler_switches_only_when_cross_view_uv_is_active(self):
        config = base_config()
        apply_subject460_stage(config, 9)
        self.assertFalse(paired_sampler_required(config))
        apply_subject460_stage(config, 10)
        self.assertTrue(paired_sampler_required(config))

    def test_seam_is_delayed_and_does_not_change_tv_coefficient(self):
        config = base_config()
        apply_subject460_stage(config, 47)
        self.assertFalse(config["exp.lf_uv.seam.enable"])
        apply_subject460_stage(config, 48)
        self.assertTrue(config["exp.lf_uv.seam.enable"])
        self.assertEqual(config["exp.lf_uv.seam.weight_pair"], .05)
        self.assertEqual(config["exp.lf_uv.seam.weight_tv"], .004)

    def test_hf_parameters_are_required_and_not_overwritten(self):
        config = base_config()
        apply_subject460_stage(config, 70)
        self.assertEqual(
            [config[f"exp.hf_img.{key}"] for key in
             ("kernel1", "sigma1", "kernel2", "sigma2", "lambda_h1", "lambda_h2")],
            [3, .8, 5, 1.2, 1., 1.],
        )
        config["exp.hf_img.sigma1"] = None
        with self.assertRaisesRegex(ValueError, "sigma1"):
            apply_subject460_stage(config, 70)

    def test_weights_are_direct_coefficients(self):
        config = base_config()
        for stage in SUBJECT460_100EP:
            apply_subject460_stage(config, stage.start)
            self.assertEqual(config["exp.lf_img.lambda"], stage.lf)
            self.assertEqual(config["exp.lf_uv.loss.weight"], stage.uv)
            self.assertEqual(config["exp.lf_uv.seam.weight_pair"], stage.seam)
            self.assertEqual(config["exp.hf_img.lambda"], stage.hf)

    def test_checkpoint_next_epoch_selects_resume_stage(self):
        config = base_config()
        stage = apply_subject460_stage(config, 48)
        self.assertEqual(stage.label, "[48,70)")
        self.assertEqual(config["exp.lf_uv.seam.weight_pair"], .05)
        stage = apply_subject460_stage(config, 70)
        self.assertEqual(stage.label, "[70,90)")
        self.assertEqual(config["exp.hf_img.lambda"], .05)

    def test_trainer_and_entrypoint_have_dynamic_wiring(self):
        trainer = (ROOT / "joint_trainer.py").read_text()
        entrypoint = (ROOT / "train.py").read_text()
        restore = (ROOT / "utils.py").read_text()
        ast.parse(trainer)
        ast.parse(entrypoint)
        self.assertIn("apply_subject460_stage", trainer)
        self.assertIn("train_loader_provider", trainer)
        self.assertIn("DynamicTrainLoader", entrypoint)
        self.assertIn("return self.current_epoch < self.config[\"training.epochs\"]", trainer)
        self.assertIn("return self.current_epoch <= self.config[\"training.epochs\"]", trainer)
        self.assertIn("self._init_seam_maps()", trainer)
        self.assertIn("require_training_state", trainer)
        self.assertIn("include_fresco=not schedule_enabled(self.config)", trainer)
        self.assertIn('required_keys = ("epoch", "global_step", "stage", "stage_step", "optimizer", "scheduler")', restore)


if __name__ == "__main__":
    unittest.main()
