from __future__ import annotations

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from misalignment.augmentation import build_augmentation, describe_augmentation


class TestImageNetAutoAugmentLevel(unittest.TestCase):
    def test_level_three_is_level_one_plus_imagenet_autoaugment(self):
        transform = build_augmentation(SimpleNamespace(), "imagenet1k", 3)
        kinds = [type(operation) for operation in transform.transforms]
        self.assertEqual(kinds[:3], [transforms.RandomResizedCrop, transforms.RandomHorizontalFlip,
                                     transforms.AutoAugment])
        self.assertEqual(transform.transforms[0].size, (224, 224))
        self.assertEqual(transform.transforms[2].policy, transforms.AutoAugmentPolicy.IMAGENET)
        self.assertEqual(transform.transforms[2].interpolation, transforms.InterpolationMode.BILINEAR)
        level_one = build_augmentation(SimpleNamespace(), "imagenet1k", 1)
        self.assertEqual(kinds[:2] + kinds[3:], [type(operation) for operation in level_one.transforms])

    def test_level_three_output_and_seeded_reproducibility(self):
        image = Image.new("RGB", (300, 260), (120, 90, 60))
        transform = build_augmentation(SimpleNamespace(), "imagenet1k", 3)
        outputs = []
        for _ in range(2):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(5)
                outputs.append(transform(image))
        self.assertEqual(tuple(outputs[0].shape), (3, 224, 224))
        self.assertTrue(torch.equal(outputs[0], outputs[1]))

    def test_level_three_metadata(self):
        metadata = describe_augmentation("imagenet1k", 3)
        self.assertEqual(metadata["input_size"], 224)
        self.assertTrue(metadata["stochastic"])
        self.assertEqual(metadata["operations"][:3], [
            "RandomResizedCrop(size=224)", "RandomHorizontalFlip(p=0.5)", "AutoAugment(policy=imagenet)",
        ])

    def test_level_three_is_imagenet1k_only(self):
        for dataset_name in ("imagenet21k_p", "cifar10"):
            with self.subTest(dataset=dataset_name):
                if dataset_name == "cifar10":
                    self.assertNotIn("AutoAugment(policy=imagenet)",
                                     describe_augmentation(dataset_name, 3)["operations"])
                else:
                    with self.assertRaisesRegex(ValueError, "supports augmentation levels"):
                        describe_augmentation(dataset_name, 3)


if __name__ == "__main__":
    unittest.main()
