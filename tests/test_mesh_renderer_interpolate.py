import unittest

import torch

from networks.meshface.mesh_renderer.mesh_renderer import _prepare_interpolate_inputs


class InterpolateInputTests(unittest.TestCase):
    def test_noncontiguous_inputs_are_made_contiguous_without_detaching(self):
        attr_base = torch.randn(2, 5, 4, requires_grad=True)
        attr = attr_base[..., :3]
        rast = torch.randn(2, 4, 6, 8)[..., ::2]
        tri = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.int32).t().contiguous().t()

        self.assertFalse(attr.is_contiguous())
        self.assertFalse(rast.is_contiguous())
        self.assertFalse(tri.is_contiguous())

        prepared_attr, prepared_rast, prepared_tri = _prepare_interpolate_inputs(
            attr, rast, tri, "test interpolation"
        )

        self.assertTrue(prepared_attr.is_contiguous())
        self.assertTrue(prepared_rast.is_contiguous())
        self.assertTrue(prepared_tri.is_contiguous())
        prepared_attr.sum().backward()
        self.assertIsNotNone(attr_base.grad)

    def test_invalid_raster_shape_names_the_tensor(self):
        attr = torch.randn(1, 5, 3)
        rast = torch.randn(1, 4, 6, 3)
        tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, r"rast.*\[B, H, W, 4\]"):
            _prepare_interpolate_inputs(attr, rast, tri, "camera-space position")

    def test_invalid_dtype_names_the_tensor(self):
        attr = torch.randn(1, 5, 3, dtype=torch.float64)
        rast = torch.randn(1, 4, 6, 4)
        tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        with self.assertRaisesRegex(TypeError, r"attr must be torch.float32"):
            _prepare_interpolate_inputs(attr, rast, tri, "camera-space position")


if __name__ == "__main__":
    unittest.main()
