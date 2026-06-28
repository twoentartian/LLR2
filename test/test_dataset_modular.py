import unittest

import numpy as np

from py_src.ml_setup_dataset.dataset_modular import ArithmeticDataset


def _dataset_members(dataset):
    return {dataset.tokenizer.decode(dataset.data[index]) for index in range(len(dataset))}


class ArithmeticDatasetSplitTests(unittest.TestCase):
    def test_chessboard_random_partition_depends_on_seed(self):
        train_a, val_a = ArithmeticDataset.splits(
            train_pct=50,
            operator="x+y_mod_17",
            modulus=17,
            train_split_type="chessboard_random",
            seed=123,
        )
        train_b, val_b = ArithmeticDataset.splits(
            train_pct=50,
            operator="x+y_mod_17",
            modulus=17,
            train_split_type="chessboard_random",
            seed=123,
        )
        train_c, val_c = ArithmeticDataset.splits(
            train_pct=50,
            operator="x+y_mod_17",
            modulus=17,
            train_split_type="chessboard_random",
            seed=124,
        )

        self.assertEqual(_dataset_members(train_a), _dataset_members(train_b))
        self.assertEqual(_dataset_members(val_a), _dataset_members(val_b))
        self.assertNotEqual(_dataset_members(train_a), _dataset_members(train_c))
        self.assertNotEqual(_dataset_members(val_a), _dataset_members(val_c))

    def test_chessboard_random_inv_flips_only_upper_right_quadrant(self):
        modulus = 7
        seed = 123
        random_train, random_val = ArithmeticDataset._get_spatial_train_val_masks(
            "x+y_mod_7",
            modulus,
            50,
            "chessboard_random",
            rng=np.random.default_rng(seed),
        )
        inverse_train, inverse_val = ArithmeticDataset._get_spatial_train_val_masks(
            "x+y_mod_7",
            modulus,
            50,
            "chessboard_random_inv",
            rng=np.random.default_rng(seed),
        )

        midpoint = modulus // 2
        upper_right = np.zeros((modulus, modulus), dtype=bool)
        upper_right[:midpoint, midpoint:] = True

        self.assertTrue((inverse_train[upper_right] == ~random_train[upper_right]).all())
        self.assertTrue((inverse_train[~upper_right] == random_train[~upper_right]).all())
        self.assertTrue((inverse_val == ~inverse_train).all())
        self.assertTrue((random_val == ~random_train).all())


if __name__ == "__main__":
    unittest.main()
