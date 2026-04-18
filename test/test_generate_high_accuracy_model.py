"""Smoke tests for model setup factories with dummy datasets.

Each test trains for exactly one batch and, for classifiers, evaluates one
validation batch. All data is synthetic and in-memory.
"""

from __future__ import annotations

import os
import sys
import unittest

from torchvision import transforms

from py_src.ml_setup import (
    cct14_7x2_imagenet1k,
    cct_7_3x1_cifar10,
    ddpm_cifar10,
    densenet121_cifar10,
    densenet_cifar_cifar10,
    dla46c_imagenet10,
    dla_cifar10,
    dla_cifar100,
    efficientnet_b0_cifar10,
    efficientnet_b0_cifar100,
    efficientnet_b1_imagenet1k,
    efficientnet_v2_s_imagenet1k,
    lenet4_mnist,
    lenet5_large_fc_mnist,
    lenet5_mnist,
    mobilenet_v2_cifar10,
    mobilenet_v2_cifar100,
    mobilenet_v3_large_imagenet1k,
    regnet_x_200mf_cifar10,
    regnet_x_200mf_cifar100,
    resnet18_cifar10,
    resnet18_imagenet1k,
    resnet50_imagenet1k,
    shufflenet_v2_cifar10,
    shufflenet_v2_cifar100,
    simplenet_cifar10,
    simplenet_cifar100,
    vgg11_bn_cifar10,
    vgg11_no_bn_cifar10,
)

# ---------------------------------------------------------------------------
# Make sure the LLR2 project root is importable regardless of where the test
# runner is launched from.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from py_src.ml_setup.ml_setup import MLSetup
from test.util import (
    make_dummy_cifar10,
    make_dummy_cifar100,
    make_dummy_imagenet10,
    make_dummy_imagenet1k,
    make_dummy_mnist,
    run_single_batch,
)


class TestRunSingleBatch(unittest.TestCase):
    """Smoke-tests: one train batch + one val batch (where applicable)."""

    def _assert_classifier_single_batch(
        self,
        name: str,
        setup: MLSetup,
        *,
        batch_size: int = 2,
        use_cpu: bool = True,
    ) -> None:
        train_result, val_result = run_single_batch(
            setup,
            run_val=True,
            batch_size=batch_size,
            use_cpu=use_cpu,
        )
        self.assertGreater(train_result.iterations, 0, name)
        self.assertGreater(train_result.total_count, 0, name)
        self.assertIsNotNone(train_result.accuracy, name)
        self.assertIsNotNone(val_result, name)
        assert val_result is not None
        self.assertGreater(val_result.total_count, 0, name)
        print(name)
        print(f"train loss:{train_result.avg_loss:.4f} train accuracy:{train_result.accuracy:.4f}")
        print(f"val loss:{val_result.avg_loss:.4f}")

    def test_ddpm_cifar10_train_no_val(self):
        setup = ddpm_cifar10(override_dataset=make_dummy_cifar10(return_pil=False))
        train_result, val_result = run_single_batch(setup, run_val=False, batch_size=2)
        self.assertGreater(train_result.iterations, 0)
        self.assertIsNone(val_result)
        print("ddpm_cifar10")
        print(f"train loss:{train_result.avg_loss:.4f}")

    def test_resnet18_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "resnet18_cifar10",
            resnet18_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=4,
        )

    def test_resnet18_imagenet1k_train_and_val_p1(self):
        self._assert_classifier_single_batch(
            "resnet18_imagenet1k_p1",
            resnet18_imagenet1k(1, override_dataset=make_dummy_imagenet1k(return_pil=False)),
            batch_size=2,
            use_cpu=False,
        )

    def test_resnet18_imagenet1k_train_and_val_p2(self):
        self._assert_classifier_single_batch(
            "resnet18_imagenet1k_p2",
            resnet18_imagenet1k(2, override_dataset=make_dummy_imagenet1k(return_pil=False)),
            batch_size=2,
            use_cpu=False,
        )

    def test_resnet50_imagenet1k_train_and_val_p2(self):
        self._assert_classifier_single_batch(
            "resnet50_imagenet1k_p2",
            resnet50_imagenet1k(2, override_dataset=make_dummy_imagenet1k(return_pil=False)),
            batch_size=1,
            use_cpu=False,
        )

    def test_cct_7_3x1_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "cct_7_3x1_cifar10",
            cct_7_3x1_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
            use_cpu=False,
        )

    def test_cct14_7x2_imagenet1k_train_and_val(self):
        self._assert_classifier_single_batch(
            "cct14_7x2_imagenet1k",
            cct14_7x2_imagenet1k(override_dataset=make_dummy_imagenet1k(return_pil=False)),
            batch_size=2,
            use_cpu=False,
        )

    def test_densenet121_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "densenet121_cifar10",
            densenet121_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_densenet_cifar_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "densenet_cifar_cifar10",
            densenet_cifar_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_dla_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "dla_cifar10",
            dla_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_dla_cifar100_train_and_val(self):
        self._assert_classifier_single_batch(
            "dla_cifar100",
            dla_cifar100(override_dataset=make_dummy_cifar100(return_pil=False)),
            batch_size=2,
        )

    def test_dla46c_imagenet10_train_and_val(self):
        self._assert_classifier_single_batch(
            "dla46c_imagenet10",
            dla46c_imagenet10(override_dataset=make_dummy_imagenet10(return_pil=False)),
            batch_size=2,
            use_cpu=False,
        )

    def test_efficientnet_b0_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "efficientnet_b0_cifar10",
            efficientnet_b0_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_efficientnet_b0_cifar100_train_and_val(self):
        self._assert_classifier_single_batch(
            "efficientnet_b0_cifar100",
            efficientnet_b0_cifar100(override_dataset=make_dummy_cifar100(return_pil=False)),
            batch_size=2,
        )

    def test_efficientnet_v2_s_imagenet1k_train_and_val(self):
        self._assert_classifier_single_batch(
            "efficientnet_v2_s_imagenet1k",
            efficientnet_v2_s_imagenet1k(override_dataset=make_dummy_imagenet1k(return_pil=False)),
            batch_size=1,
            use_cpu=False,
        )

    def test_efficientnet_b1_imagenet1k_train_and_val(self):
        self._assert_classifier_single_batch(
            "efficientnet_b1_imagenet1k",
            efficientnet_b1_imagenet1k(override_dataset=make_dummy_imagenet1k(return_pil=False)),
            batch_size=1,
            use_cpu=False,
        )

    def test_lenet5_mnist_train_and_val(self):
        self._assert_classifier_single_batch(
            "lenet5_mnist",
            lenet5_mnist(override_dataset=make_dummy_mnist(return_pil=False)),
            batch_size=4,
        )

    def test_lenet4_mnist_train_and_val(self):
        self._assert_classifier_single_batch(
            "lenet4_mnist",
            lenet4_mnist(override_dataset=make_dummy_mnist(return_pil=False)),
            batch_size=4,
        )

    def test_lenet5_large_fc_mnist_train_and_val(self):
        self._assert_classifier_single_batch(
            "lenet5_large_fc_mnist",
            lenet5_large_fc_mnist(override_dataset=make_dummy_mnist(return_pil=False)),
            batch_size=4,
        )

    def test_mobilenet_v2_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "mobilenet_v2_cifar10",
            mobilenet_v2_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_mobilenet_v2_cifar100_train_and_val(self):
        self._assert_classifier_single_batch(
            "mobilenet_v2_cifar100",
            mobilenet_v2_cifar100(override_dataset=make_dummy_cifar100(return_pil=False)),
            batch_size=2,
        )

    def test_mobilenet_v3_large_imagenet1k_train_and_val(self):
        self._assert_classifier_single_batch(
            "mobilenet_v3_large_imagenet1k",
            mobilenet_v3_large_imagenet1k(2, override_dataset=make_dummy_imagenet1k(return_pil=False)),
            batch_size=2,
            use_cpu=False,
        )

    def test_regnet_x_200mf_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "regnet_x_200mf_cifar10",
            regnet_x_200mf_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_regnet_x_200mf_cifar100_train_and_val(self):
        self._assert_classifier_single_batch(
            "regnet_x_200mf_cifar100",
            regnet_x_200mf_cifar100(override_dataset=make_dummy_cifar100(return_pil=False)),
            batch_size=2,
        )

    def test_shufflenet_v2_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "shufflenet_v2_cifar10",
            shufflenet_v2_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_shufflenet_v2_cifar100_train_and_val(self):
        self._assert_classifier_single_batch(
            "shufflenet_v2_cifar100",
            shufflenet_v2_cifar100(override_dataset=make_dummy_cifar100(return_pil=False)),
            batch_size=2,
        )

    def test_simplenet_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "simplenet_cifar10",
            simplenet_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_simplenet_cifar100_train_and_val(self):
        self._assert_classifier_single_batch(
            "simplenet_cifar100",
            simplenet_cifar100(override_dataset=make_dummy_cifar100(return_pil=False)),
            batch_size=2,
        )

    def test_vgg11_bn_cifar10_train_and_val(self):
        self._assert_classifier_single_batch(
            "vgg11_bn_cifar10",
            vgg11_bn_cifar10(override_dataset=make_dummy_cifar10(return_pil=False)),
            batch_size=2,
        )

    def test_vgg11_no_bn_cifar10_train_and_val(self):
        resize_224 = [transforms.Resize((224, 224))]
        self._assert_classifier_single_batch(
            "vgg11_no_bn_cifar10",
            vgg11_no_bn_cifar10(
                override_dataset=make_dummy_cifar10(
                    return_pil=False,
                    transforms_training=resize_224,
                    transforms_testing=resize_224,
                )
            ),
            batch_size=1,
        )


if __name__ == "__main__":
    unittest.main()
