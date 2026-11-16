import unittest
from pathlib import Path

from exp.config_utils import normalize_dotted_config, validate_hf_image_config


ROOT = Path(__file__).resolve().parents[1]


class HFImageConfigTests(unittest.TestCase):
    @staticmethod
    def _dotted_hf_values(path):
        result = {}
        for line in path.read_text().splitlines():
            content = line.split("#", 1)[0].strip()
            if not content.startswith("exp.hf_img.") or ":" not in content:
                continue
            key, value = (part.strip() for part in content.split(":", 1))
            try:
                result[key] = float(value) if "." in value else int(value)
            except ValueError:
                result[key] = value.strip('"\'')
        return result

    def test_full_config_has_explicit_paper_hf_parameters(self):
        config = self._dotted_hf_values(ROOT / "configs/nersemble/460/full.yaml")
        values = validate_hf_image_config(config)
        self.assertEqual(values["kernel1"], 3)
        self.assertEqual(values["sigma1"], 0.8)
        self.assertEqual(values["kernel2"], 5)
        self.assertEqual(values["sigma2"], 1.2)
        self.assertEqual(values["lambda_h1"], 1.0)
        self.assertEqual(values["lambda_h2"], 1.0)

    def test_missing_hf_parameters_name_every_missing_key(self):
        config = self._dotted_hf_values(ROOT / "configs/nersemble/460/full_J1J2.yaml")
        with self.assertRaisesRegex(
            ValueError, r"exp\.hf_img\.sigma1.*exp\.hf_img\.sigma2"
        ):
            validate_hf_image_config(config, context="test")

    def test_none_is_missing_not_a_numeric_default(self):
        config = {
            "exp.hf_img.kernel1": 3,
            "exp.hf_img.sigma1": None,
            "exp.hf_img.kernel2": 5,
            "exp.hf_img.sigma2": 1.2,
            "exp.hf_img.lambda_h1": 1.0,
            "exp.hf_img.lambda_h2": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "exp.hf_img.sigma1"):
            validate_hf_image_config(config)

    def test_nested_values_are_available_after_normalization(self):
        config = {"exp": {"hf_img": {
            "kernel1": 3, "sigma1": 0.8, "kernel2": 5, "sigma2": 1.2,
            "lambda_h1": 1.0, "lambda_h2": 1.0,
        }}}
        values = validate_hf_image_config(normalize_dotted_config(config))
        self.assertEqual(values["sigma1"], 0.8)
        self.assertEqual(values["sigma2"], 1.2)


if __name__ == "__main__":
    unittest.main()
