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

    def test_chessboard_random_inv_flips_only_upper_right_half(self):
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

        rows, cols = np.indices((modulus, modulus))
        upper_right = cols > rows

        self.assertTrue((inverse_train[upper_right] == ~random_train[upper_right]).all())
        self.assertTrue((inverse_train[~upper_right] == random_train[~upper_right]).all())
        self.assertTrue((inverse_val == ~inverse_train).all())
        self.assertTrue((random_val == ~random_train).all())

    def test_chessboard_random_transpose_ratio_preserves_partition_sizes(self):
        modulus = 17
        seed = 123
        symmetric_train, _ = ArithmeticDataset._get_spatial_train_val_masks(
            "x+y_mod_17",
            modulus,
            50,
            "chessboard_random",
            rng=np.random.default_rng(seed),
        )
        ratio_train, ratio_val = ArithmeticDataset._get_spatial_train_val_masks(
            "x+y_mod_17",
            modulus,
            50,
            "chessboard_random",
            rng=np.random.default_rng(seed),
            chessboard_transpose_ratio=80,
        )

        expected_mismatch_pairs = round((1 - 0.8) * modulus * modulus / 2)
        expected_agreement_count = modulus * modulus - 2 * expected_mismatch_pairs

        self.assertEqual(ratio_train.sum(), symmetric_train.sum())
        self.assertEqual((ratio_train == ratio_train.T).sum(), expected_agreement_count)
        self.assertTrue((ratio_val == ~ratio_train).all())

        train_dataset, val_dataset = ArithmeticDataset.splits(
            train_pct=50,
            operator="x+y_mod_7",
            modulus=7,
            train_split_type="chessboard_random",
            seed=seed,
            chessboard_transpose_ratio=80,
        )
        partition_by_operands = {}
        for dataset, partition in ((train_dataset, "train"), (val_dataset, "val")):
            for index in range(len(dataset)):
                tokens = dataset.tokenizer.decode(dataset.data[index]).split()
                partition_by_operands[(int(tokens[1]), int(tokens[3]))] = partition

        actual_agreement_count = sum(
            partition == partition_by_operands[(b, a)]
            for (a, b), partition in partition_by_operands.items()
        )
        expected_dataset_agreement = 7 * 7 - 2 * round((1 - 0.8) * 7 * 7 / 2)
        self.assertEqual(actual_agreement_count, expected_dataset_agreement)
        self.assertIn("_transpose80_", train_dataset.name)

    def test_chessboard_random_default_remains_fully_symmetric(self):
        train, _ = ArithmeticDataset._get_spatial_train_val_masks(
            "x+y_mod_17",
            17,
            50,
            "chessboard_random",
            rng=np.random.default_rng(123),
        )

        self.assertTrue((train == train.T).all())

    def test_chessboard_transpose_ratio_validation(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            ArithmeticDataset.splits(
                train_pct=50,
                operator="x+y_mod_17",
                modulus=17,
                train_split_type="chessboard_random",
                chessboard_transpose_ratio=101,
            )

        with self.assertRaisesRegex(ValueError, "only be changed"):
            ArithmeticDataset.splits(
                train_pct=50,
                operator="x+y_mod_17",
                modulus=17,
                train_split_type="random",
                chessboard_transpose_ratio=80,
            )


if __name__ == "__main__":
    unittest.main()
