import unittest

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


if __name__ == "__main__":
    unittest.main()
