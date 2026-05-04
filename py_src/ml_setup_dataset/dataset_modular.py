import itertools
import math
import time
import unittest
from datetime import datetime
from typing import List, Dict, Union, Optional, Literal

from concurrent.futures import ProcessPoolExecutor

import torch
from torch import Tensor, LongTensor
import numpy as np
from tqdm import tqdm
from sympy.combinatorics.permutations import Permutation
from mod import Mod

import blobfile as bf


VALID_OPERATORS = {
    "+": "addition",
    "-": "subtraction",
    "*": "muliplication",
    "/": "division",
    "**2+": "squarepoly",
    "**3+": "cubepoly",
    "x**2+y**2_mod_97": "quad1",
    "x**2+y**2+x*y_mod_97": "quad2",
    "x**2+y**2+x*y+x_mod_97": "quad3",
    "x**3+x*y_mod_97": "cube1",
    "x**3+x*y**2+y_mod_97": "cube2",
    "(x._value//y)if(y._value%2==1)else(x-y)_mod_97": "mix1",
    "s5": "s5",
    "s5conj": "s5conj",
    "s5aba": "s5aba",
    "+*": "even-addition_odd-multiplication",
    "+-": "even-addition_odd-subtraction",
    "sort": "sort",
    "reverse": "reverse",
    "copy": "copy",
    "unknown": "unknown",
}
EOS_TOKEN = "<|eos|>"
EQ_TOKEN = "="
# MODULUS = 97
# NUMS = list(range(MODULUS))

def render(operand, join_str=""):
    if (
        isinstance(operand, list)
        or isinstance(operand, tuple)
        or isinstance(operand, np.ndarray)
    ):
        return join_str.join(map(render, operand))
    elif isinstance(operand, Permutation):
        return "".join(map(str, operand.array_form))
    elif isinstance(operand, Mod):
        return str(operand._value)
    else:
        return str(operand)


class ArithmeticTokenizer:
    """Stores the list of token text to token id mappings and converts between them"""
    def __init__(self, modulus) -> None:
        self.itos = self.get_tokens(modulus)
        self.stoi: Dict[str, int] = dict([(s, i) for i, s in enumerate(self.itos)])

    def _encode(self, s: str) -> Tensor:
        tokens = [self.stoi[t] if t in self.stoi.keys() else self.stoi["unknown"] for t in s.split(" ")]
        return LongTensor(tokens)

    def encode(self, obj: Union[str, List]) -> Tensor:
        """
        Convert a string of text into a rank-1 tensor of token ids
        or convert a list of strings of text into a rank-2 tensor of token ids

        :param obj: the string or list of strings to convert
        :returns: a tensor of the token ids
        """
        if isinstance(obj, str):
            return self._encode(obj)
        elif isinstance(obj, list):
            if len(obj) == 0:
                return torch.empty((0, 0), dtype=torch.long)
            return torch.stack([self._encode(s) for s in obj], dim=0)
        else:
            raise NotImplementedError

    def decode(self, tensor: Tensor, with_brackets: bool = False) -> str:
        """
        Convert a tensor of token ids into a string of text

        :param tensor: a tensor of the token ids
        :param with_brackets: if true, the returned string will include <> brackets
                              around the text corresponding to each token.
        :returns: string of these tokens.
        """
        indices = tensor.long()
        if with_brackets:
            l = "<"
            r = ">"
        else:
            l = ""
            r = ""
        tokens = [l + self.itos[i] + r for i in indices]
        return " ".join(tokens)

    def __len__(self) -> int:
        """
        :returns: the number of tokens in this vocabulary
        """
        return len(self.itos)

    def save_tokens(self, file_path: str):
        """
        Save the token vocabulary to a file.
        This is necessary for loading datasets later.

        :param file_path: Path to save token file
        """

        # Ensure directory exists
        bf.makedirs(bf.dirname(file_path))

        # Write tokens to file
        with bf.BlobFile(file_path, "w") as f:
            f.write("\n".join(self.itos))

        return file_path

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        Load a tokenizer from a saved token vocabulary file.

        :param file_path: Path to the token file
        :returns: ArithmeticTokenizer instance with loaded vocabulary
        """
        # Read tokens from file
        with bf.BlobFile(file_path, "r") as f:
            tokens = f.read().strip().split("\n")

        # Create a new tokenizer instance without calling __init__
        tokenizer = cls.__new__(cls)
        tokenizer.itos = tokens
        tokenizer.stoi = dict([(s, i) for i, s in enumerate(tokens)])

        return tokenizer

    @classmethod
    def get_tokens(cls, modulus):
        nums = list(range(modulus))
        tokens = (
            [EOS_TOKEN, EQ_TOKEN]
            + list(map(render, nums,))
            + list(map(render, itertools.permutations(range(5))))  # s5
            + list(sorted(list(VALID_OPERATORS.keys())))
        )
        return tokens


class ArithmeticDataset:
    """A Dataset of arithmetic equations"""

    @classmethod
    def splits(
        cls,
        train_pct: float,
        operator: str,
        modulus = 97,
        operand_length: Optional[int] = None,
        train_split_type: Literal["random", "chessboard", "updown", "leftright", "tl_to_br", "tr_to_bl", "interlace_row", "interlace_col", "chessboard_random"] = "random",
    ):
        """
        Creates training and validation datasets

        :param train_pct: percentage of total equations used for training data
        :param operator: The arithmetic operator for this dataset e.g. '+', '-', '*', '/', 'sort'
        :param operand_length: for list based datasets the length of the lists
        :returns: (train_dataset, validation_dataset)
        """

        assert (0 < train_pct) and (train_pct <= 100)

        ds_name = cls.get_dsname(modulus, operator, operand_length, train_pct, train_split_type)
        eqs_train, eqs_val = cls.make_data(operator, modulus, operand_length, train_split_type=train_split_type, train_pct=train_pct)

        train_ds = cls(ds_name, eqs_train, modulus, train=True)
        val_ds = cls(ds_name, eqs_val, modulus, train=False)

        return train_ds, val_ds

    @classmethod
    def calc_split_len(cls, train_pct, ds_len):
        train_rows = round(ds_len * (train_pct / 100.0))
        val_rows = ds_len - train_rows
        return train_rows, val_rows

    def __init__(self, name, data: Union[Tensor, List[str]], modulus, train, tokenizer=None) -> None:
        """
        :param data: A list of equations strings. Each equation must have an '=' in it.
        """
        self.tokenizer = ArithmeticTokenizer(modulus) if tokenizer is None else tokenizer
        self.modulus = modulus
        self.name = name
        self.train = train
        if isinstance(data, list):
            self.data = self.tokenizer.encode(data)
        else:
            self.data = data

    def __len__(self) -> int:
        """
        :returns: total number of equations in this dataset
        """
        return self.data.shape[0]

    def get_first_data_tensor(self):
        return self.data[0]

    def save_to_file(self, filepath: str) -> str:
        """
        Save the dataset to a human-readable text file.

        :param filepath: Path to save the file. If None, uses default naming based on dataset name.
        :param save_tokenizer: If True, also saves the tokenizer vocabulary file (tokens.txt)
        :returns: The filepath where the dataset was saved
        """

        # Ensure directory exists
        bf.makedirs(bf.dirname(filepath))

        # Decode all equations to human-readable format
        equations = []
        for i in range(len(self.data)):
            eq_tokens = self.data[i]
            eq_str = self.tokenizer.decode(eq_tokens)
            equations.append(eq_str)

        # Write to file
        with bf.BlobFile(filepath, "w") as f:
            f.write("\n".join(equations))

        return filepath

    @classmethod
    def load_from_file(cls, filepath: str, modulus: int, name: Optional[str] = None, train: bool = True, tokenizer_path=None):
        """
        Load a dataset from a text file.

        :param filepath: Path to the text file containing equations
        :param name: Name for the dataset. If None, derived from filename
        :param train: Whether this is a training dataset
        :param data_dir: Directory containing tokens.txt. If None, uses directory of filepath
        :returns: ArithmeticDataset instance
        """
        if name is None:
            # Extract name from filepath
            name = bf.basename(filepath).replace(".txt", "")

        tokenizer = None
        if tokenizer_path is not None:
            tokenizer = ArithmeticTokenizer.load_from_file(tokenizer_path)

        # Read equations from file
        with bf.BlobFile(filepath, "r") as f:
            raw_text = f.read()
        stripped = raw_text.strip()
        equations = [] if stripped == "" else stripped.split("\n")

        print(f"Loaded {len(equations)} equations from {filepath}")
        return cls(name, equations, modulus, train, tokenizer=tokenizer)

    # @classmethod
    # def _render(cls, operand):
    #    return render(operand, join_str=" ")
    #
    # @classmethod
    # def _render_eq(parts):
    #    return " ".join(map(render, parts))

    @classmethod
    def _make_binary_operation_data(cls, operator: str, modulus, operands=None) -> tuple[List[str], dict[tuple[int, int], str]]:
        nums = list(range(modulus))
        if operator == "s5":
            operands = operands or list(range(5))
            elems = map(np.array, itertools.permutations(operands))
            tuples = itertools.product(elems, repeat=2)
        elif operator in ["s5conj", "s5aba"]:
            operands = operands or list(range(5))
            elems = map(Permutation, itertools.permutations(operands))
            tuples = itertools.product(elems, repeat=2)
        elif "_mod_" in operator:
            elems = [Mod(i, modulus) for i in range(modulus)]
            tuples = itertools.product(elems, repeat=2)
        else:
            operands = operands or nums
            tuples = itertools.product(operands, repeat=2)

        # if operator == "s5":
        #     print("elems", list(elems))
        #     print("tuples", list(tuples))
        eqs = []
        eqs_table = dict()
        for a, b in tuples:
            if operator == "/":
                if b == 0:
                    continue
                else:
                    c = a
                    a = (b * c) % modulus
            elif operator == "s5":
                c = b[a] # type: ignore
            elif operator == "s5conj":
                c = a * b * (a.__invert__()) # type: ignore
            elif operator == "s5aba":
                c = a * b * a
            elif operator == "+*":
                if a % 2 == 0: # type: ignore
                    c = (a + b) % modulus
                else:
                    c = (a * b) % modulus
            elif operator == "+-":
                if a % 2 == 0: # type: ignore
                    c = (a + b) % modulus
                else:
                    c = (a - b) % modulus
            elif "_mod_" in operator:
                items = operator.split("_mod_")
                expression = items[0]
                modulo = int(items[-1])
                function = eval(f"lambda x, y: ({expression}) % {modulo}")
                c = function(int(a), int(b))
            else:
                c = eval(f"({a} {operator} {b}) % {modulus}")
            eq = " ".join(map(render, [a, operator, b, "=", c]))
            eqs.append(eq)
            eqs_table[(a, b)] = eq

        # if operator == "s5":
        #     print("eqs", eqs)
        return eqs, eqs_table

    # @staticmethod
    # def _render_unop_example(operator, lhs, rhs):
    #    return " ".join([operator, render(lhs), "=", render(rhs)])

    @staticmethod
    def _make_unary_operation_data(operator: str, operands: Tensor) -> List[str]:
        """
        :param operator: The unary operator to apply to each operand e.g. '+'
        :param operands: A tensor of operands
        :returns: list of equations"""
        num_examples = len(operands)

        if operator == "sort":
            rhs = torch.sort(operands, dim=1)[0]
        elif operator == "reverse":
            rhs = torch.flip(operands, dims=(1,))
        elif operator == "copy":
            rhs = operands
        else:
            raise Exception("unsupported operator")

        def func(L, R):
            L = map(str, L)
            R = map(str, R)
            return f"{operator} {' '.join(L)} = {' '.join(R)}"

        if num_examples < 1000000000:
            eqs = [
                func(L, R)
                for L, R in tqdm(
                    zip(operands.tolist(), rhs.tolist()), total=num_examples
                )
            ]
        else:
            with ProcessPoolExecutor() as executor:
                eqs = executor.map(func, tqdm(zip(operands, rhs), total=num_examples))

        return eqs # type: ignore

    # @staticmethod
    # def _make_s5_data(abstract=False) -> List[str]:
    #    elems = itertools.permutations([0, 1, 2, 3, 4])
    #    pairs = itertools.product(elems, repeat=2)
    #    eqs = []
    #    for a, b in pairs:
    #        a = np.array(a)
    #        b = np.array(b)
    #        c = b[a]
    #        eq = " ".join(map(render, (a, "s5", b, "=", c)))
    #        eq = cls._render_eq([a, , b, "=", c])
    #        eqs.append(eq)
    #
    #    return eqs

    @classmethod
    def get_dsname(cls, modulus, operator, operand_length, train_pct, split_type) -> str:
        operator, noise_level = cls._get_operator_and_noise_level(operator)
        if operator in VALID_OPERATORS:
            ds_name = f"modulus{modulus}_{VALID_OPERATORS[operator]}_train{train_pct}_{split_type}"
        else:
            ds_name = f"modulus{modulus}_{operator}_train{train_pct}_{split_type}"
        if operand_length is not None:
            ds_name += f"_{operand_length}"
        if noise_level > 0:
            ds_name += f"_noise{noise_level}"
        ds_name += datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        ds_name = ds_name.replace("**", "^")
        ds_name = ds_name.replace("*", "")
        return ds_name

    @classmethod
    def _get_operator_and_noise_level(cls, operator):
        if "_noisy" in operator:
            operator, noise_level = operator.split("_noisy_")
            return operator, int(noise_level)
        else:
            return operator, 0

    @classmethod
    def make_data(cls, operator, modulus, operands=None, shuffle=True, seed=None, train_split_type="random", train_pct: float = 0.5) -> tuple[List[str], List[str]]:
        operator, noise_level = cls._get_operator_and_noise_level(operator)
        data, data_table = None, None
        if operator not in ["sort", "reverse", "copy"]:
            data, data_table = cls._make_binary_operation_data(operator, modulus)
        else:
            data = cls._make_unary_operation_data(operator, operands) # type: ignore

        if seed is None:
            seed = time.time_ns()
        rng = np.random.default_rng(seed)

        if train_split_type != "random":
            assert train_pct is not None, ("train_pct must be provided for spatial train_split_type")
            assert data_table is not None, ("Spatial splits are only supported for binary-operation datasets")
            assert noise_level == 0, ("noise level has to be 0 for non-random splits")
            train_mask, val_mask = cls._get_spatial_train_val_masks(operator, modulus, train_pct, train_split_type)

            elems_a = list(range(modulus))
            elems_b = list(range(modulus))

            n = len(elems_a)
            train_eqs, val_eqs = [], []
            for i in range(n):
                for j in range(n):
                    a, b = elems_a[i], elems_b[j]
                    key = (a, b)
                    # data_table may use the raw operand objects as keys
                    # for integer operators the keys are plain ints
                    eq = data_table.get(key) or data_table.get(
                        (elems_a[i], elems_b[j])
                    )
                    if eq is None:
                        continue  # e.g. division by zero was skipped
                    eq_wrapped = EOS_TOKEN + " " + eq + " " + EOS_TOKEN
                    if train_mask[i, j]:
                        train_eqs.append(eq_wrapped)
                    else:
                        val_eqs.append(eq_wrapped)
            if len(val_eqs) == 0:
                val_eqs.append(train_eqs[0])
            if shuffle:
                rng.shuffle(train_eqs)
                rng.shuffle(val_eqs)

            return train_eqs, val_eqs
        else:
            data = [EOS_TOKEN + " " + eq + " " + EOS_TOKEN for eq in data]
            if shuffle:
                rng.shuffle(data)
            if noise_level > 0:
                random_answer_eqns = rng.choice(data, size=noise_level)
                random_answers = [
                    random_eq.split(" = ")[1] for random_eq in random_answer_eqns
                ]
                for i in range(noise_level):
                    data[i] = data[i].split(" = ")[0] + " = " + random_answers[i]

            train_rows, _ = cls.calc_split_len(train_pct, len(data))
            train_eqs = data[:train_rows]
            val_eqs = data[train_rows:]

            if len(val_eqs) == 0:
                val_eqs.append(train_eqs[0])

            return train_eqs, val_eqs



    @classmethod
    def _get_spatial_train_val_masks(cls, operator, modulus, train_pct, train_split_type):
        """
        Build boolean train/val masks over the n×n grid of (a, b) operand pairs.

        The grid has shape (n, n) where n = modulus (or len(permutations) for s5).
        Row index = first operand a, column index = second operand b.

        Returns
        -------
        train_mask : np.ndarray of bool, shape (n, n)
        val_mask   : np.ndarray of bool, shape (n, n)
        """
        if operator in ["s5", "s5conj", "s5aba"]:
            import math as _math
            n = _math.factorial(5)  # 120
        elif "_mod_" in operator:
            n = int(operator.split("_mod_")[-1])
        else:
            n = modulus

        frac = train_pct / 100.0
        rows = np.arange(n)
        cols = np.arange(n)
        i, j = np.meshgrid(rows, cols, indexing="ij")  # i=row (a), j=col (b)

        if train_split_type == "chessboard":
            train_mask = (i + j) % 2 == 0

        elif train_split_type == "updown":
            # Top rows → train, bottom rows → val
            cutoff = round(n * frac)
            train_mask = i < cutoff

        elif train_split_type == "leftright":
            # Left cols → train, right cols → val
            cutoff = round(n * frac)
            train_mask = j < cutoff

        elif train_split_type == "tl_to_br":
            # Triangle from top-left: train where i + j < cutoff diagonal
            # Total cells = n*n, train cells ≈ frac * n*n
            # For the triangle sum_{d=0}^{D-1} min(d+1, n) cells; use simple threshold on i+j
            total = n * n
            target_train = round(total * frac)
            # Count cells with i+j <= threshold
            diag_sum = i + j
            # Find threshold T such that #{i+j <= T} ~= target_train
            counts = np.array([(diag_sum <= T).sum() for T in range(2 * n - 1)])
            T = int(np.searchsorted(counts, target_train, side="left"))
            train_mask = diag_sum <= T

        elif train_split_type == "tr_to_bl":
            # Triangle from top-right: train where (n-1-j) + i < cutoff diagonal
            # Equivalent to selecting from top-right corner sweeping to bottom-left
            diag_sum = i + (n - 1 - j)
            total = n * n
            target_train = round(total * frac)
            counts = np.array([(diag_sum <= T).sum() for T in range(2 * n - 1)])
            T = int(np.searchsorted(counts, target_train, side="left"))
            train_mask = diag_sum <= T

        elif train_split_type == "interlace_row":
            train_mask = i % 2 == 0  # even rows → train, odd rows → val

        elif train_split_type == "interlace_col":
            train_mask = j % 2 == 0  # even cols → train, odd cols → val

        elif train_split_type == "chessboard_random":
            M = ((i + j) % 2 == 0).astype(np.int8)

            rng = np.random.default_rng(int(frac * 1e9))  # reproducible
            num_swaps = n * n * 50  # enough passes for thorough mixing
            max_attempts = num_swaps * 20

            swaps_done = 0
            for _ in range(max_attempts):
                if swaps_done >= num_swaps:
                    break
                # Sample 4 distinct indices to form a rectangle
                idx = rng.choice(n, size=4, replace=False)
                r0, r1, c0, c1 = int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3])
                # Check anti-diagonal pattern in the 2x2 block
                if M[r0, c0] == 1 and M[r1, c1] == 1 and M[r0, c1] == 0 and M[r1, c0] == 0:
                    # Swap the 2x2 block
                    M[r0, c0] = 0
                    M[r0, c1] = 1
                    M[r1, c0] = 1
                    M[r1, c1] = 0
                    # Mirror the same swap across the diagonal (transpose indices)
                    M[c0, r0] = 0
                    M[c1, r0] = 1
                    M[c0, r1] = 1
                    M[c1, r1] = 0
                    swaps_done += 1

            train_mask = M.astype(bool)

        else:
            raise ValueError(f"Unknown train_split_type: {train_split_type}")

        val_mask = ~train_mask
        return train_mask, val_mask


class ArithmeticIterator(torch.utils.data.IterableDataset):
    """
    An iterator over batches of data in an ArithmeticDataset
    """

    def __init__(
        self,
        dataset: ArithmeticDataset,
        device: torch.device,
        batchsize_hint: float|int = 0,
        shuffle: bool = True,
    ) -> None:
        """
        :param dataset: the dataset to iterate over
        :param device: the torch device to send batches to
        :param batchsize_hint: * 0 means we use a default batchsize
                               * -1 means the entire dataset
                               * float between 0 and 1 means each batch is
                                 that fraction of the DS
                               * int > 1 means that specific batch size
        :param shuffle: whether or not to randomly shuffle the dataset
        """
        self.dataset = dataset
        self.batchsize = self.calculate_batchsize(
            len(dataset), batchsize_hint=batchsize_hint # type: ignore
        )
        self.device = device
        self.reset_iteration(shuffle=shuffle)

    @staticmethod
    def calculate_batchsize(ds_size: int, batchsize_hint: int|float = 0) -> int:
        """
        Calculates which batch size to use

        :param ds_size: the number of equations in the dataset
        :param batchsize_hint: * 0 means we use a default batchsize
                               * -1 means the entire dataset
                               * float between 0 and 1 means each batch is
                                 that fraction of the DS
                               * int > 1 means that specific batch size
        :returns: the actual batchsize to use
        """

        if batchsize_hint == -1:
            return ds_size
        elif batchsize_hint == 0:
            return min(512, math.ceil(ds_size / 2.0))
        elif (batchsize_hint > 0) and (batchsize_hint < 1):
            return math.ceil(ds_size * batchsize_hint)
        elif batchsize_hint > 1:
            assert isinstance(batchsize_hint, int)
            return min(batchsize_hint, ds_size)
        else:
            raise ValueError("batchsize_hint must be >= -1")

    def reset_iteration(self, shuffle=True):
        self.index = 0
        if shuffle and self.dataset.train:
            self.permutation = torch.randperm(len(self.dataset))
        else:
            self.permutation = torch.arange(len(self.dataset))

    def __iter__(self):
        """
        :returns: this iterator
        """
        return self

    def __next__(self) -> Dict[str, Tensor]:
        """
        Returns one batch of data.

        :raises: StopIteration when we're out of data
        :returns: batch tensor of shape (self.batchsize, tokens_per_eq)
        """

        batch_begin = self.index * self.batchsize
        if batch_begin > len(self.dataset) - 1:
            self.reset_iteration()
            raise StopIteration
        indices = self.permutation[batch_begin : batch_begin + self.batchsize]
        text = self.dataset.data[indices, :-1]
        target = self.dataset.data[indices, 1:]
        batch = {"text": text.to(self.device), "target": target.to(self.device)}
        self.index += 1
        return batch

    def __len__(self) -> int:
        """
        :returns: the total number of batches
        """
        return math.ceil(len(self.dataset) / self.batchsize)

class TestStringMethods(unittest.TestCase):
    def test_generate_dataset(self):
        (train_dataset, val_dataset,) = ArithmeticDataset.splits(
            train_pct=50,  # type: ignore
            operator="+",  # type: ignore
        )
        iterator = ArithmeticIterator(
            train_dataset,
            torch.device("cpu"),
            batchsize_hint=10,  # type: ignore
        )

        for idx, batch in enumerate(iterator):
            tok = train_dataset.tokenizer
            text = tok.decode(batch["text"][0])
            print(f"text: {text}")
            target = tok.decode(batch["target"][0])
            print(f"target: {target}")

    def test_example_1_basic_with_tokenizer(self):
        test_data_folder_name = "test_data"
        modulus = 97

        """Example showing tokenizer is saved automatically"""
        print("=" * 70)
        print("Generating datasets...")
        print("=" * 70)

        # Generate datasets
        train_dataset, val_dataset = ArithmeticDataset.splits(
            train_pct=80,
            operator="+",
            train_split_type="chessboard_random",
        )

        print(f"\nGenerated {len(train_dataset)} train examples")
        print(f"Generated {len(val_dataset)} validation examples")

        # Save datasets (tokenizer saved automatically)
        print("\n" + "=" * 70)
        print("Saving datasets...")
        print("=" * 70)

        train_file = train_dataset.save_to_file(f"./{test_data_folder_name}/{train_dataset.name}/train.txt")
        val_file = val_dataset.save_to_file(f"./{test_data_folder_name}/{val_dataset.name}/val.txt")
        tokenizer_file = train_dataset.tokenizer.save_tokens(f"./{test_data_folder_name}/{val_dataset.name}/tokenizer.txt")

        # Now tokens.txt exists in ./my_data/ directory
        print("\nFiles created:")
        print(f"  - {train_file}")
        print(f"  - {val_file}")
        print(f"  - {tokenizer_file} (tokenizer vocabulary)")

        # Load them back
        print("\n" + "=" * 70)
        print("Loading datasets...")
        print("=" * 70)

        loaded_train = ArithmeticDataset.load_from_file(
            f"{train_file}",
            modulus,
            train=True,
        )

        loaded_val = ArithmeticDataset.load_from_file(
            f"{val_file}",
            modulus,
            train=False,
        )

        # Verify
        print("\n" + "=" * 70)
        print("Verification:")
        print("=" * 70)
        print(f"Train datasets match: {torch.equal(train_dataset.data, loaded_train.data)}")
        print(f"Val datasets match: {torch.equal(val_dataset.data, loaded_val.data)}")

    def test_example_2_s5(self):
        test_data_folder_name = "test_data"
        modulus = 97

        """Example showing tokenizer is saved automatically"""
        print("=" * 70)
        print("Generating datasets...")
        print("=" * 70)

        # Generate datasets
        train_dataset, val_dataset = ArithmeticDataset.splits(
            train_pct=50,
            operator="s5",
        )

        print(f"\nGenerated {len(train_dataset)} train examples")
        print(f"Generated {len(val_dataset)} validation examples")

        # Save datasets (tokenizer saved automatically)
        print("\n" + "=" * 70)
        print("Saving datasets...")
        print("=" * 70)

        train_file = train_dataset.save_to_file(f"./{test_data_folder_name}/{train_dataset.name}/train.txt")
        val_file = val_dataset.save_to_file(f"./{test_data_folder_name}/{val_dataset.name}/val.txt")
        tokenizer_file = train_dataset.tokenizer.save_tokens(f"./{test_data_folder_name}/{val_dataset.name}/tokenizer.txt")

        # Now tokens.txt exists in ./my_data/ directory
        print("\nFiles created:")
        print(f"  - {train_file}")
        print(f"  - {val_file}")
        print(f"  - {tokenizer_file} (tokenizer vocabulary)")

        # Load them back
        print("\n" + "=" * 70)
        print("Loading datasets...")
        print("=" * 70)

        loaded_train = ArithmeticDataset.load_from_file(
            f"{train_file}",
            modulus,
            train=True,
        )

        loaded_val = ArithmeticDataset.load_from_file(
            f"{val_file}",
            modulus,
            train=False,
        )

        # Verify
        print("\n" + "=" * 70)
        print("Verification:")
        print("=" * 70)
        print(f"Train datasets match: {torch.equal(train_dataset.data, loaded_train.data)}")
        print(f"Val datasets match: {torch.equal(val_dataset.data, loaded_val.data)}")

    def test_example_3_noisy(self):
        test_data_folder_name = "test_data"
        modulus = 97

        """Example showing tokenizer is saved automatically"""
        print("=" * 70)
        print("Generating datasets...")
        print("=" * 70)

        # Generate datasets
        train_dataset, val_dataset = ArithmeticDataset.splits(
            train_pct=50,
            operator="+_noisy_10",
        )

        print(f"\nGenerated {len(train_dataset)} train examples")
        print(f"Generated {len(val_dataset)} validation examples")

        # Save datasets (tokenizer saved automatically)
        print("\n" + "=" * 70)
        print("Saving datasets...")
        print("=" * 70)

        train_file = train_dataset.save_to_file(f"./{test_data_folder_name}/{train_dataset.name}/train.txt")
        val_file = val_dataset.save_to_file(f"./{test_data_folder_name}/{val_dataset.name}/val.txt")
        tokenizer_file = train_dataset.tokenizer.save_tokens(f"./{test_data_folder_name}/{val_dataset.name}/tokenizer.txt")

        # Now tokens.txt exists in ./my_data/ directory
        print("\nFiles created:")
        print(f"  - {train_file}")
        print(f"  - {val_file}")
        print(f"  - {tokenizer_file} (tokenizer vocabulary)")

        # Load them back
        print("\n" + "=" * 70)
        print("Loading datasets...")
        print("=" * 70)

        loaded_train = ArithmeticDataset.load_from_file(
            f"{train_file}",
            modulus,
            train=True,
        )

        loaded_val = ArithmeticDataset.load_from_file(
            f"{val_file}",
            modulus,
            train=False,
        )

        # Verify
        print("\n" + "=" * 70)
        print("Verification:")
        print("=" * 70)
        print(f"Train datasets match: {torch.equal(train_dataset.data, loaded_train.data)}")
        print(f"Val datasets match: {torch.equal(val_dataset.data, loaded_val.data)}")
