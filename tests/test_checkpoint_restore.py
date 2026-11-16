import unittest

import torch

from networks.gshair.hairwrapper import GSHairWrapper
from networks.meshface.facewrapper import MeshFaceWrapper


class _Logger:
    def info(self, *args, **kwargs):
        pass


class _FailingModel:
    def __init__(self, reason):
        self.reason = reason
        self.called = False

    def state_dict(self):
        return {"weight": torch.zeros(1)}

    def load_state_dict(self, *args, **kwargs):
        self.called = True
        raise ValueError(self.reason)


class _TrackingModel:
    def __init__(self):
        self.called = False

    def state_dict(self):
        return {"weight": torch.zeros(1)}

    def load_state_dict(self, *args, **kwargs):
        self.called = True


class CheckpointRestorePropagationTests(unittest.TestCase):
    def _hair_wrapper(self, model):
        wrapper = object.__new__(GSHairWrapper)
        torch.nn.Module.__init__(wrapper)
        wrapper.models = {"deform_mlp": model}
        wrapper.cfg = {"gs.upSH": 1000}
        return wrapper

    def _face_wrapper(self, model):
        wrapper = object.__new__(MeshFaceWrapper)
        wrapper.models = {"head_mlp": model}
        return wrapper

    def test_hair_load_failure_propagates_model_and_original_reason(self):
        model = _FailingModel("synthetic hair failure")
        wrapper = self._hair_wrapper(model)
        with self.assertRaisesRegex(
            RuntimeError, "deform_mlp.*synthetic hair failure"
        ) as caught:
            wrapper.restore_models(
                {"deform_mlp": {"weight": torch.ones(1)}},
                optimizer=None,
                global_step=0,
                logger=_Logger(),
            )
        self.assertTrue(model.called)
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_hair_missing_optional_key_is_skipped(self):
        model = _TrackingModel()
        wrapper = self._hair_wrapper(model)
        wrapper.restore_models({}, optimizer=None, global_step=0, logger=_Logger())
        self.assertFalse(model.called)

    def test_face_load_failure_propagates_model_and_original_reason(self):
        model = _FailingModel("synthetic face failure")
        wrapper = self._face_wrapper(model)
        with self.assertRaisesRegex(
            RuntimeError, "head_mlp.*synthetic face failure"
        ) as caught:
            wrapper.restore_models(
                {"head_mlp": {"weight": torch.ones(1)}}, logger=_Logger()
            )
        self.assertTrue(model.called)
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_face_missing_optional_key_is_skipped(self):
        model = _TrackingModel()
        wrapper = self._face_wrapper(model)
        wrapper.restore_models({}, logger=_Logger())
        self.assertFalse(model.called)


if __name__ == "__main__":
    unittest.main()
