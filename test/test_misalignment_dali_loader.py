from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
import misalignment_measurement_2 as measurement
from misalignment.augmentation import describe_augmentation
from misalignment.dali_loader import (
    DaliImageNetLoader, _RandomEraseSource, image_file_manifest, resolve_loader_backend,
)


class MetadataOnlyImages(Dataset):
    def __init__(self, prefix, count):
        self.samples = [(f"/{prefix}/{index}.jpg", index % 3) for index in range(count)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raise AssertionError("Extracting a manifest must not load/decode images")


class TestDaliLoaderMetadata(unittest.TestCase):
    def test_auto_backend_and_explicit_pytorch_override(self):
        self.assertEqual(resolve_loader_backend("imagenet1k"), "dali")
        for name in ("mnist", "cifar10", "cifar100", "modular"):
            self.assertEqual(resolve_loader_backend(name), "pytorch")
            with self.assertRaisesRegex(ValueError, "only for imagenet1k"):
                resolve_loader_backend(name, "dali")
        self.assertEqual(resolve_loader_backend("imagenet1k", "pytorch"), "pytorch")
        with self.assertRaises(ValueError):
            resolve_loader_backend("imagenet1k", "invalid")

    def test_cli_default_and_override(self):
        for flags, expected in (([], "auto"), (["--loader_backend", "pytorch"], "pytorch"),
                                (["--loader-backend", "dali"], "dali")):
            with patch.object(sys, "argv", ["test", "--dataset", "imagenet1k", *flags]):
                self.assertEqual(measurement.parse_args().loader_backend, expected)

    def test_nested_manifest_preserves_split_permutation_without_image_reads(self):
        original = ConcatDataset([MetadataOnlyImages("train", 7), MetadataOnlyImages("val", 4)])
        augmented = measurement.AugmentedDataset(original, None, 11, False)
        partition_a, partition_b, permutation, excluded = measurement._split_dataset(
            augmented, seed=123, odd_size_policy="drop",
        )
        all_samples = original.datasets[0].samples + original.datasets[1].samples
        for partition, indices in ((partition_a, permutation[:5]), (partition_b, permutation[5:10])):
            files, labels = image_file_manifest(partition)
            self.assertEqual(list(zip(files, labels)), [all_samples[index] for index in indices.tolist()])
        self.assertEqual(len(excluded), 1)
        files_a, _ = image_file_manifest(partition_a)
        files_b, _ = image_file_manifest(partition_b)
        self.assertFalse(set(files_a) & set(files_b))
        nested = Subset(partition_a, [4, 0])
        self.assertEqual(image_file_manifest(nested)[0], [files_a[4], files_a[0]])

    def test_manifest_requires_file_metadata(self):
        with self.assertRaisesRegex(TypeError, "ImageFolder samples"):
            image_file_manifest(TensorDataset(torch.ones(3, 2)))
        with self.assertRaisesRegex(ValueError, "empty partition"):
            image_file_manifest(MetadataOnlyImages("empty", 0))

    def test_cpu_dali_rejected_without_loading_images(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            measurement._build_loader(
                MetadataOnlyImages("data", 3), 2, device=torch.device("cpu"), num_workers=4,
                seed=1, loader_backend="dali", augmentation_config=describe_augmentation("imagenet1k", 0),
            )

    def test_missing_dali_has_explicit_fallback_hint(self):
        from misalignment.dali_loader import _import_dali

        with patch.dict(sys.modules, {"nvidia.dali": None}):
            with self.assertRaisesRegex(ImportError, "--loader_backend pytorch"):
                _import_dali()

    def test_erase_rectangles_are_reproducible_valid_and_do_not_use_torch_rng(self):
        before = torch.random.get_rng_state().clone()
        source_a = _RandomEraseSource(1000, 224, 123)
        source_b = _RandomEraseSource(1000, 224, 123)
        anchors_a, shapes_a = source_a()
        anchors_b, shapes_b = source_b()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        np.testing.assert_array_equal(anchors_a, anchors_b)
        np.testing.assert_array_equal(shapes_a, shapes_b)
        erased = 0
        for anchor, shape in zip(anchors_a, shapes_a):
            self.assertEqual(shape.dtype, np.int32)
            self.assertTrue(np.all(anchor >= 0))
            self.assertTrue(np.all(anchor + shape <= 224))
            if shape.sum():
                erased += 1
                self.assertTrue(np.all((shape > 0) & (shape < 224)))
        self.assertTrue(60 < erased < 140)


def has_dali_gpu():
    try:
        return torch.cuda.is_available() and importlib.util.find_spec("nvidia.dali") is not None
    except ModuleNotFoundError:
        return False


@unittest.skipUnless(has_dali_gpu(), "NVIDIA DALI and CUDA required for real pipeline tests")
class TestDaliLoaderGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        # Two original splits, eleven differently sized JPEGs, two classes.
        for split, count in (("train", 7), ("val", 4)):
            for name in ("c0", "c1"):
                (cls.root / split / name).mkdir(parents=True)
            for index in range(count):
                height, width = 235 + 3 * index, 281 - 2 * index
                y, x = np.indices((height, width))
                pixels = np.stack(((x + index * 29) % 256, (y * 2 + index * 17) % 256,
                                   (x + y + index * 11) % 256), axis=-1).astype(np.uint8)
                Image.fromarray(pixels).save(cls.root / split / f"c{index % 2}" / f"{index}.jpg")
        cls.original = ConcatDataset([ImageFolder(cls.root / "train", transform=None),
                                      ImageFolder(cls.root / "val", transform=None)])

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def make_partitions(self, level):
        augmented = measurement.AugmentedDataset(self.original, None, 123, level > 0)
        return measurement._split_dataset(augmented, seed=11, odd_size_policy="drop")

    def make_loader(self, partition, level, seed=123, input_size=None):
        return measurement._build_loader(
            partition, 2, device=torch.device("cuda"), num_workers=2, seed=seed,
            prefetch_factor=2, loader_backend="dali",
            augmentation_config=describe_augmentation("imagenet1k", level, input_size=input_size),
        )

    def read(self, loader):
        batches = list(loader)
        return torch.cat([batch[0] for batch in batches]), torch.cat([batch[1] for batch in batches]), batches

    def test_all_augmentation_levels_gpu_shape_labels_and_repeated_passes(self):
        for level in (0, 1, 2):
            with self.subTest(level=level):
                partition_a, _, _, _ = self.make_partitions(level)
                loader = self.make_loader(partition_a, level)
                try:
                    before_cpu = torch.random.get_rng_state().clone()
                    before_cuda = torch.cuda.get_rng_state().clone()
                    images, labels, batches = self.read(loader)
                    crop_size = 176 if level == 2 else 224
                    self.assertEqual(images.shape, (5, 3, crop_size, crop_size))
                    self.assertEqual(images.dtype, torch.float32)
                    self.assertEqual(images.device.type, "cuda")
                    self.assertEqual(labels.device.type, "cuda")
                    self.assertEqual(labels.dtype, torch.int64)
                    self.assertEqual([batch[1].shape for batch in batches], [(2,), (2,), (1,)])
                    self.assertEqual(labels.tolist(), image_file_manifest(partition_a)[1])
                    self.assertTrue(torch.isfinite(images).all().item())
                    repeated, repeated_labels, _ = self.read(loader)
                    self.assertTrue(torch.equal(images, repeated))
                    self.assertTrue(torch.equal(labels, repeated_labels))
                    self.assertTrue(torch.equal(before_cpu, torch.random.get_rng_state()))
                    self.assertTrue(torch.equal(before_cuda, torch.cuda.get_rng_state()))
                    self.assertEqual(len(loader), 3)
                finally:
                    loader.close()

    def test_interrupted_pass_and_epoch_rewind_restart_at_first_sample(self):
        for level in (0, 1, 2):
            with self.subTest(level=level):
                partition_a, _, _, _ = self.make_partitions(level)
                loader = self.make_loader(partition_a, level, input_size=224)
                try:
                    images, labels, _ = self.read(loader)
                    iterator = iter(loader)
                    first = next(iterator)
                    self.assertTrue(torch.equal(first[0], images[:2]))
                    iterator.close()
                    repeated, repeated_labels, _ = self.read(loader)
                    self.assertTrue(torch.equal(images, repeated))
                    self.assertTrue(torch.equal(labels, repeated_labels))
                    measurement._set_augmentation_epoch(partition_a, 1)
                    changed, _, _ = self.read(loader)
                    self.assertEqual(changed.shape, (5, 3, 224, 224))
                    if level > 0:
                        self.assertFalse(torch.equal(images, changed))
                    measurement._set_augmentation_epoch(partition_a, 0)
                    rewound, _, _ = self.read(loader)
                    self.assertTrue(torch.equal(images, rewound))
                finally:
                    loader.close()

    def test_none_preprocessing_close_to_torchvision_and_swapped_half_order(self):
        partition_a, partition_b, _, _ = self.make_partitions(0)
        loader_a, loader_b = self.make_loader(partition_a, 0, 101), self.make_loader(partition_b, 0, 202)
        try:
            images_a, labels_a, _ = self.read(loader_a)
            images_b, labels_b, _ = self.read(loader_b)
            transform = measurement.build_augmentation(None, "imagenet1k", "none")
            reference = torch.stack([transform(Image.open(path).convert("RGB"))
                                     for path in image_file_manifest(partition_a)[0]]).cuda()
            # JPEG/resize implementations are not pixelwise equal. Large
            # differences would reveal wrong mean/std/crop/interpolation.
            self.assertLess((images_a - reference).abs().mean().item(), 0.04)
            loader_a.close()
            loader_b.close()
            swapped_b, swapped_labels_b, _ = self.read(loader_b)
            swapped_a, swapped_labels_a, _ = self.read(loader_a)
            self.assertTrue(torch.equal(images_a, swapped_a))
            self.assertTrue(torch.equal(images_b, swapped_b))
            self.assertTrue(torch.equal(labels_a, swapped_labels_a))
            self.assertTrue(torch.equal(labels_b, swapped_labels_b))
        finally:
            loader_a.close()
            loader_b.close()

    def test_limited_flatness_restarts_and_close_releases_pipeline(self):
        partition_a, _, _, _ = self.make_partitions(0)
        loader = self.make_loader(partition_a, 0)
        model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2)).cuda()
        options = dict(device=torch.device("cuda"), modular=False, eq_position=None,
                       layer_names=["2.weight"], hutchinson_samples=2, seed=71)
        try:
            limited = measurement.measure_relative_flatness(model, loader, nn.CrossEntropyLoss(), max_batches=1, **options)
            self.assertEqual(limited[0]["examples"], 2)
            complete = measurement.measure_relative_flatness(model, loader, nn.CrossEntropyLoss(), max_batches=-1, **options)
            self.assertEqual(complete[0]["examples"], 5)
            repeated = measurement.measure_relative_flatness(model, loader, nn.CrossEntropyLoss(), max_batches=-1, **options)
            self.assertEqual(complete, repeated)
            pipeline_reference = weakref.ref(loader._pipeline)
            loader.close()
            self.assertIsNone(pipeline_reference())
        finally:
            loader.close()

    def test_real_training_both_directions_checkpoint_flatness_and_metadata(self):
        partition_a, partition_b, permutation, excluded = self.make_partitions(0)
        model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))
        bundle = measurement.DatasetBundle(
            partition_a, partition_b,
            SimpleNamespace(model=model, criterion=nn.CrossEntropyLoss(), model_type=SimpleNamespace(name="toy"),
                            dataset_type="imagenet1k"),
            "imagenet1k", "imagenet1k", 11, permutation, excluded, False,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            flags = ["test", "--dataset", "imagenet1k", "--epochs", "2", "--batch_size", "2",
                     "--relative_flatness_batch_size", "2", "--relative_flatness_samples", "2",
                     "--num_workers", "2", "--prefetch_factor", "2", "--random_seed", "123",
                     "--learning_rate", "0.001", "--warmup_epochs", "0", "--no-compile", "--amp",
                     "--output_folder_name", str(output)]
            with patch.object(sys, "argv", flags), patch.object(measurement, "load_dataset_bundle", return_value=bundle):
                measurement.main()
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["loader_backend"], "dali")
            self.assertEqual(len(report["runs"]), 2)
            for run in report["runs"].values():
                self.assertEqual(run["final"]["train"]["examples"], 5)
                self.assertEqual(run["final"]["val"]["examples"], 5)
                self.assertTrue(math.isfinite(run["misalignment_measure"]["overall_mean"]))
                self.assertEqual(run["runtime"]["loader"]["train"]["backend"], "dali")
                self.assertTrue(run["runtime"]["evaluation_amp"])
                self.assertFalse(run["runtime"]["relative_flatness_amp"])
                for point in run["gap_checkpoints"]["union"]:
                    self.assertEqual(point["relative_flatness"]["partitions"]["train"][0]["examples"], 5)
            checkpoint = output / report["runs"]["a_as_train_b_as_val"]["files"]["final_model"]
            checkpoint_output = output / "checkpoint_only"
            checkpoint_flags = flags[:-2] + ["--flatness_from_checkpoint", str(checkpoint),
                                             "--output_folder_name", str(checkpoint_output)]
            with patch.object(sys, "argv", checkpoint_flags), \
                 patch.object(measurement, "load_dataset_bundle", return_value=bundle), \
                 patch.object(measurement, "build_optimizer_and_scheduler") as optimizer_factory:
                measurement.main()
                optimizer_factory.assert_not_called()
            checkpoint_report = json.loads((checkpoint_output / "flatness_from_checkpoint.json").read_text())
            self.assertEqual(checkpoint_report["loader"]["partition_a"]["backend"], "dali")
            self.assertEqual(checkpoint_report["relative_flatness"]["partitions"]["a"][0]["examples"], 5)


if __name__ == "__main__":
    unittest.main()
