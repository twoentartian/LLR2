from __future__ import annotations

import unittest

import numpy as np

from py_src.ml_setup_model.nanoclip.nanoclip import NanoCLIP


class TestNanoCLIPValidation(unittest.TestCase):
    def test_calculate_recall_uses_numpy_compatible_membership_check(self) -> None:
        recall = NanoCLIP._calculate_recall(
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([0, 1]),
            k_values=[1],
        )
        self.assertEqual(recall.shape, (1,))
        self.assertEqual(float(recall[0]), 1.0)

    def test_validation_epoch_end_stores_scalar_loss_and_correct_count(self) -> None:
        model = NanoCLIP.__new__(NanoCLIP)
        object.__setattr__(model, "validation_descriptors", {
            "img": [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)],
            "txt": [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)],
            "nb_captions": [1],
        })
        object.__setattr__(model, "latest_loss", None)
        object.__setattr__(model, "latest_correct_count", None)

        NanoCLIP.on_validation_epoch_end(model)
        loss, correct_count = model.get_validation_result()

        self.assertIsInstance(loss, float)
        self.assertEqual(loss, 1.0)
        self.assertEqual(correct_count, 2)


if __name__ == "__main__":
    unittest.main()
