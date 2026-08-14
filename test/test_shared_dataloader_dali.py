from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model import ModelType
from py_src.shared_dataloader import BatchRequest
from py_src.shared_dataloader_dali import (
    DaliSharedDataLoader,
    ensure_supported_dali_workload,
    is_supported_dali_workload,
)


def _dali_cuda_available() -> bool:
    try:
        import nvidia.dali  # noqa: F401
        import nvidia.dali.plugin.pytorch  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


class _RawDataset:
    def __init__(self, data: np.ndarray, targets: list[int]):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = torch.from_numpy(self.data[index])
        if image.ndim == 3:
            image = image.permute(2, 0, 1)
        else:
            image = image.unsqueeze(0)
        return image.to(dtype=torch.float32) / 255.0, self.targets[index]


def _setup(model_type: ModelType, dataset_type: DatasetType):
    return SimpleNamespace(
        model_type=model_type,
        dataset_type=dataset_type,
        collate_fn=None,
    )


class DaliWorkloadSelectionTest(unittest.TestCase):
    def test_only_requested_workloads_are_supported(self):
        supported = (
            (ModelType.lenet4, DatasetType.mnist),
            (ModelType.lenet5, DatasetType.mnist),
            (ModelType.mobilenet_v2, DatasetType.cifar10),
            (ModelType.resnet18_bn, DatasetType.cifar10),
            (ModelType.resnet18_bn, DatasetType.cifar100),
            (ModelType.cct_7_3x1_32, DatasetType.cifar10),
        )
        for model_type, dataset_type in supported:
            with self.subTest(model=model_type.name, dataset=dataset_type.name):
                setup = _setup(model_type, dataset_type)
                self.assertTrue(is_supported_dali_workload(setup))
                ensure_supported_dali_workload(setup)

        unsupported = (
            (ModelType.lenet4, DatasetType.cifar10),
            (ModelType.mobilenet_v2, DatasetType.cifar100),
            (ModelType.resnet18_gn, DatasetType.cifar10),
            (ModelType.cct_7_3x1_32, DatasetType.cifar100),
        )
        for model_type, dataset_type in unsupported:
            with self.subTest(model=model_type.name, dataset=dataset_type.name):
                setup = _setup(model_type, dataset_type)
                self.assertFalse(is_supported_dali_workload(setup))
                with self.assertRaisesRegex(ValueError, "not supported"):
                    ensure_supported_dali_workload(setup)


@unittest.skipUnless(
    _dali_cuda_available(),
    "NVIDIA DALI and CUDA are required for DALI shared-loader tests",
)
class DaliSharedDataLoaderTest(unittest.TestCase):
    def test_mnist_is_cached_and_routed_on_gpu(self):
        rng = np.random.default_rng(7)
        dataset = _RawDataset(
            rng.integers(0, 256, size=(12, 28, 28), dtype=np.uint8),
            list(range(12)),
        )
        loader = DaliSharedDataLoader(
            dataset,
            _setup(ModelType.lenet4, DatasetType.mnist),
            device_id=0,
            num_threads=1,
            seed=19,
        )
        try:
            self.assertEqual(loader.raw_images.device, torch.device("cuda:0"))
            self.assertEqual(loader.raw_images.dtype, torch.uint8)
            self.assertEqual(tuple(loader.raw_images.shape), (12, 28, 28, 1))

            loader.set_plan(
                [
                    BatchRequest.from_indices(3, 0, [1, 4, 7, 9]),
                    BatchRequest.from_indices(8, 0, [2, 5]),
                ]
            )
            batches = list(loader)
        finally:
            loader.close()

        self.assertEqual(
            [(batch.node_name, batch.batch_index) for batch in batches],
            [(3, 0), (8, 0)],
        )
        first_images, first_targets = batches[0].batch
        second_images, second_targets = batches[1].batch
        self.assertEqual(tuple(first_images.shape), (4, 1, 28, 28))
        self.assertEqual(tuple(second_images.shape), (2, 1, 28, 28))
        self.assertEqual(first_images.dtype, torch.float32)
        self.assertEqual(first_images.device, torch.device("cuda:0"))
        self.assertTrue(torch.isfinite(first_images).all().item())
        self.assertEqual(first_targets.tolist(), [1, 4, 7, 9])
        self.assertEqual(second_targets.tolist(), [2, 5])

    def test_cifar10_uses_gpu_augmentation_and_normalization(self):
        rng = np.random.default_rng(11)
        dataset = _RawDataset(
            rng.integers(0, 256, size=(10, 32, 32, 3), dtype=np.uint8),
            list(range(10)),
        )
        loader = DaliSharedDataLoader(
            dataset,
            _setup(ModelType.mobilenet_v2, DatasetType.cifar10),
            device_id=0,
            num_threads=1,
            seed=23,
        )
        try:
            loader.set_plan([BatchRequest.from_indices(4, 2, [0, 3, 6, 9])])
            routed = next(iter(loader))
        finally:
            loader.close()

        images, targets = routed.batch
        self.assertEqual((routed.node_name, routed.batch_index), (4, 2))
        self.assertEqual(tuple(images.shape), (4, 3, 32, 32))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(images.device, torch.device("cuda:0"))
        self.assertTrue(torch.isfinite(images).all().item())
        self.assertEqual(targets.tolist(), [0, 3, 6, 9])

    def test_cifar100_pipeline_supports_resnet_output_labels(self):
        rng = np.random.default_rng(13)
        dataset = _RawDataset(
            rng.integers(0, 256, size=(8, 32, 32, 3), dtype=np.uint8),
            [0, 17, 33, 49, 65, 81, 98, 99],
        )
        loader = DaliSharedDataLoader(
            dataset,
            _setup(ModelType.resnet18_bn, DatasetType.cifar100),
            device_id=0,
            num_threads=1,
            seed=29,
        )
        try:
            loader.set_plan([BatchRequest.from_indices(1, 0, [1, 6, 7])])
            images, targets = next(iter(loader)).batch
        finally:
            loader.close()

        self.assertEqual(tuple(images.shape), (3, 3, 32, 32))
        self.assertEqual(targets.tolist(), [17, 98, 99])

    def test_simulator_factory_selects_dali_loader(self):
        import simulator
        from py_src.engine import Device
        from py_src.ml_setup import lenet4_mnist
        from py_src.ml_setup_dataset import DatasetSetup
        from py_src.ml_setup_dataset.dataset_intermediate_layer import (
            DatasetWithFastLabelSelection,
        )
        from py_src.node import ensure_ml_setup_compatibility

        rng = np.random.default_rng(17)
        dataset = _RawDataset(
            rng.integers(0, 256, size=(6, 28, 28), dtype=np.uint8),
            [0, 1, 2, 3, 4, 5],
        )
        setup = ensure_ml_setup_compatibility(
            lenet4_mnist(
                override_dataset=DatasetSetup(
                    DatasetType.mnist,
                    dataset,
                    dataset,
                )
            )
        )
        selector = DatasetWithFastLabelSelection(dataset, setup)
        config = SimpleNamespace(
            preset_shared_training_loader_workers=0,
            preset_dali_num_threads=1,
            preset_dali_seed=43,
        )

        loader = simulator._create_shared_training_loader(
            selector,
            setup,
            config,
            Device("cuda:0"),
            use_dali=True,
            dali_device_id=0,
        )
        try:
            self.assertIsInstance(loader, DaliSharedDataLoader)
            loader.set_plan([BatchRequest.from_indices(0, 0, [0, 1])])
            images, targets = next(iter(loader)).batch
        finally:
            loader.close()

        self.assertEqual(tuple(images.shape), (2, 1, 28, 28))
        self.assertEqual(targets.tolist(), [0, 1])

    def test_invalid_raw_shape_and_indices_are_rejected(self):
        bad_dataset = _RawDataset(
            np.zeros((4, 27, 28), dtype=np.uint8),
            [0, 1, 2, 3],
        )
        with self.assertRaisesRegex(ValueError, "MNIST DALI input"):
            DaliSharedDataLoader(
                bad_dataset,
                _setup(ModelType.lenet5, DatasetType.mnist),
                device_id=0,
            )

        rng = np.random.default_rng(31)
        dataset = _RawDataset(
            rng.integers(0, 256, size=(4, 28, 28), dtype=np.uint8),
            [0, 1, 2, 3],
        )
        loader = DaliSharedDataLoader(
            dataset,
            _setup(ModelType.lenet5, DatasetType.mnist),
            device_id=0,
            num_threads=1,
        )
        try:
            with self.assertRaisesRegex(IndexError, "out-of-range"):
                loader.set_plan([BatchRequest.from_indices(0, 0, [0, 4])])
        finally:
            loader.close()

    def test_all_requested_model_dataset_pairs_train_one_batch(self):
        from py_src.engine import Device, train
        from py_src.ml_setup import (
            cct_7_3x1_cifar10,
            lenet4_mnist,
            lenet5_mnist,
            mobilenet_v2_cifar10,
            resnet18_cifar10,
            resnet18_cifar100,
        )
        from py_src.ml_setup_dataset import DatasetSetup

        workloads = (
            ("lenet4_mnist", lenet4_mnist, DatasetType.mnist, (4, 28, 28), 10),
            ("lenet5_mnist", lenet5_mnist, DatasetType.mnist, (4, 28, 28), 10),
            (
                "mobilenetv2_cifar10",
                mobilenet_v2_cifar10,
                DatasetType.cifar10,
                (4, 32, 32, 3),
                10,
            ),
            (
                "resnet18_cifar10",
                resnet18_cifar10,
                DatasetType.cifar10,
                (4, 32, 32, 3),
                10,
            ),
            (
                "resnet18_cifar100",
                resnet18_cifar100,
                DatasetType.cifar100,
                (4, 32, 32, 3),
                100,
            ),
            (
                "cct7_cifar10",
                cct_7_3x1_cifar10,
                DatasetType.cifar10,
                (4, 32, 32, 3),
                10,
            ),
        )

        for workload_name, setup_fn, dataset_type, shape, class_count in workloads:
            with self.subTest(workload=workload_name):
                rng = np.random.default_rng(37)
                dataset = _RawDataset(
                    rng.integers(0, 256, size=shape, dtype=np.uint8),
                    [index % class_count for index in range(shape[0])],
                )
                setup = setup_fn(
                    override_dataset=DatasetSetup(
                        dataset_type,
                        dataset,
                        dataset,
                    )
                )
                loader = DaliSharedDataLoader(
                    dataset,
                    setup,
                    device_id=0,
                    num_threads=1,
                    seed=41,
                )
                try:
                    loader.set_plan(
                        [BatchRequest.from_indices(0, 0, [0, 1])]
                    )
                    optimizer = torch.optim.SGD(
                        setup.model.parameters(),
                        lr=0.001,
                    )
                    result = train(
                        setup.adapter,
                        (routed.batch for routed in loader),
                        optimizer,
                        device=Device("cuda:0"),
                    )
                finally:
                    loader.close()

                self.assertEqual(result.iterations, 1)
                self.assertTrue(np.isfinite(result.avg_loss))
                if workload_name == "resnet18_cifar100":
                    self.assertEqual(setup.model.fc.out_features, 100)
                del optimizer, setup, loader
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
