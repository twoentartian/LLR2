#!/usr/bin/env python3
"""Prepare nested modular datasets and write a phase-diagram run script.

This tool deliberately does *not* run training.  It creates a copy of the
complete (master) dataset, creates fixed-label operand prefixes from it, and
writes ``run_all.py`` plus ``run_all.sh``.  The generated Python runner calls
``generate_grokking_phase_diagram.py`` once for every prefix.

For a master ``x**y mod M`` dataset, a prefix of size ``n`` contains the
examples with ``x < n`` and ``y < n`` while keeping the original labels
modulo ``M``.  The directory name contains ``modulus<n>`` because the existing
dataset loader uses that part of the name to select the dataset's operand
domain; the copied tokenizer still contains the master vocabulary.

The default scaling is based on the number of training batches:
``epoch(n) = max(2000, ceil(base_epoch / batch_count(n) ** 2))``, where
``batch_count(n) = ceil(train_examples(n) / batchsize(n))`` and
``batchsize(n) = min(n*n, 65536)``.  Both policies are command-line
configurable and the generated manifest records the resolved values for every
dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PHASE_SCRIPT = SCRIPT_DIR / "generate_grokking_phase_diagram.py"
# These are the sizes used by the existing nested sweep and reference output.
# ``--sizes`` can be used for a denser sequence (for example, every 100).
DEFAULT_SIZES = (97, 197, 297, 397, 497, 597, 697, 797, 897, 997, 1997)
DEFAULT_BASE_SIZE = 97
DEFAULT_BASE_EPOCH = 150_000
DEFAULT_EPOCH_POWER = 1.5
MIN_EPOCH = 2_000
DEFAULT_BATCHSIZE_CAP = 65_536


def parse_xy(line: str, source: Path | str = "<input>", line_number: int = 0) -> tuple[int, int]:
    """Extract x and y from the serialized equation format."""

    fields = line.split()
    try:
        equal_index = fields.index("=")
        # EOS, x, operator, y, =, rhs, EOS
        return int(fields[1]), int(fields[equal_index - 1])
    except (ValueError, IndexError) as exc:
        location = f"{source}:{line_number}" if line_number else str(source)
        raise ValueError(f"Could not parse {location}: {line!r}") from exc


def validate_master(master: Path) -> tuple[Path, Path, Path]:
    paths = (master / "train.txt", master / "val.txt", master / "tokenizer.txt")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required master file: {path}")
    return paths


def copy_master(source: Path, destination: Path) -> Path:
    """Copy the master dataset into the experiment directory if needed."""

    validate_master(source)
    destination.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    destination = destination.resolve()
    if source != destination:
        for filename in ("train.txt", "val.txt", "tokenizer.txt"):
            src = source / filename
            dst = destination / filename
            # copyfile also replaces a stale generated file, which is useful
            # when a master dataset was regenerated between experiments.
            shutil.copyfile(src, dst)
    return destination


def generate_master_dataset(
    output_root: Path,
    *,
    expression: str,
    modulus: int,
    train_pct: float,
    split_type: str,
    operand_length: int | None,
    random_seed: int | None,
) -> Path:
    """Generate a complete master dataset through the repository API."""

    # Import lazily: preparing prefixes from an existing master should not
    # require importing torch.
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from generate_grokking_phase_diagram import generate_dataset

    generated_parent = output_root / "_generated_master"
    generated_parent.mkdir(parents=True, exist_ok=True)
    train_ds, _ = generate_dataset(
        str(generated_parent),
        train_pct,
        expression,
        modulus,
        split_type,
        operand_length,
        seed=random_seed,
    )
    generated = Path(train_ds.name)
    if not generated.is_absolute():
        generated = generated_parent / generated
    if not generated.is_dir():
        # ``ArithmeticDataset.name`` is normally the generated directory name,
        # but accepting the basename makes this robust across older versions.
        generated = generated_parent / Path(train_ds.name).name
    validate_master(generated)
    return generated


def dataset_dir_name(size: int, fixed_label_modulus: int, train_pct: float, split_type: str) -> str:
    train_text = f"{train_pct:g}"
    return f"modulus{size}_x^y_mod_{fixed_label_modulus}_train{train_text}_{split_type}_prefix"


def result_dir_name(size: int, fixed_label_modulus: int, train_pct: float, split_type: str) -> str:
    train_text = f"{train_pct:g}"
    return f"x^y_mod_{fixed_label_modulus}_size{size}_train{train_text}_{split_type}_phase_diagram"


def _write_prefixes_for_file(
    source: Path,
    destinations: dict[int, Path],
    sizes: Sequence[int],
) -> tuple[int, dict[int, int]]:
    """Filter one master split into all prefixes in one streaming pass."""

    total = 0
    kept = {size: 0 for size in sizes}
    handles = {}
    try:
        for size in sizes:
            destinations[size].parent.mkdir(parents=True, exist_ok=True)
            handles[size] = destinations[size].open("w", encoding="utf-8", newline="\n")
        with source.open("r", encoding="utf-8") as infile:
            for line_number, raw_line in enumerate(infile, 1):
                line = raw_line.rstrip("\r\n")
                if not line.strip():
                    continue
                total += 1
                x, y = parse_xy(line, source, line_number)
                for size in sizes:
                    if x < size and y < size:
                        handles[size].write(line)
                        handles[size].write("\n")
                        kept[size] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return total, kept


@dataclass(frozen=True)
class DatasetPlan:
    size: int
    dataset_size: int
    batchsize: int
    batch_count: int
    epoch: int


def resolve_batchsize(size: int, *, cap: int = DEFAULT_BATCHSIZE_CAP) -> int:
    """Use the whole n-by-n dataset, capped at 65536 by default."""

    if size <= 0:
        raise ValueError("size must be positive")
    if cap <= 0:
        raise ValueError("batchsize cap must be positive")
    return min(size * size, cap)


def resolve_batch_count(size: int, batchsize: int, *, examples: int | None = None) -> int:
    """Return the number of training batches for one prefix.

    When ``examples`` is omitted, ``size * size`` is used for compatibility
    with callers that do not have a train/validation split available.
    """

    if size <= 0:
        raise ValueError("size must be positive")
    if batchsize <= 0:
        raise ValueError("batchsize must be positive")
    example_count = size * size if examples is None else examples
    if example_count <= 0:
        raise ValueError("examples must be positive")
    return max(1, math.ceil(example_count / batchsize))


def resolve_epoch(
    size: int,
    *,
    base_size: int = DEFAULT_BASE_SIZE,
    base_epoch: int = DEFAULT_BASE_EPOCH,
    power: float = DEFAULT_EPOCH_POWER,
    batchsize: int | None = None,
    batchsize_cap: int = DEFAULT_BATCHSIZE_CAP,
    train_examples: int | None = None,
) -> int:
    """Scale epochs by batch count, with a 2000 minimum.

    ``base_size`` is retained for compatibility with the earlier size-based
    API; batch-count scaling does not use it.
    """

    if size <= 0 or base_size <= 0 or base_epoch <= 0:
        raise ValueError("size, base_size and base_epoch must be positive")
    if power < 0:
        raise ValueError("epoch scaling power must be non-negative")
    effective_batchsize = (
        resolve_batchsize(size, cap=batchsize_cap)
        if batchsize is None
        else min(batchsize, size * size)
    )
    batch_count = resolve_batch_count(size, effective_batchsize, examples=train_examples)
    return max(MIN_EPOCH, math.ceil(base_epoch / (batch_count**power)))


def make_plan(
    size: int,
    *,
    batchsize_cap: int,
    base_size: int,
    base_epoch: int,
    epoch_power: float,
    batchsize_override: int | None = None,
    epoch_override: int | None = None,
    train_examples: int | None = None,
) -> DatasetPlan:
    if epoch_override is not None and epoch_override < MIN_EPOCH:
        raise ValueError(f"epoch must be at least {MIN_EPOCH}, got {epoch_override}")
    dataset_size = size * size
    resolved_batchsize = (
        resolve_batchsize(size, cap=batchsize_cap)
        if batchsize_override is None
        else min(batchsize_override, dataset_size)
    )
    batch_count = resolve_batch_count(size, resolved_batchsize, examples=train_examples)
    return DatasetPlan(
        size=size,
        dataset_size=dataset_size,
        batchsize=resolved_batchsize,
        batch_count=batch_count,
        epoch=(
            resolve_epoch(
                size,
                base_epoch=base_epoch,
                power=epoch_power,
                batchsize=resolved_batchsize,
                train_examples=train_examples,
            )
            if epoch_override is None
            else epoch_override
        ),
    )


def parse_size_overrides(values: Sequence[str], name: str, *, minimum: int = 1) -> dict[int, int]:
    """Parse repeatable ``SIZE=VALUE`` overrides used by the CLI."""

    result: dict[int, int] = {}
    for item in values:
        try:
            size_text, value_text = item.split("=", 1)
            size, value = int(size_text), int(value_text)
        except ValueError as exc:
            raise ValueError(f"{name} must use SIZE=VALUE, got {item!r}") from exc
        if size <= 0 or value < minimum:
            if minimum == 1:
                raise ValueError(f"{name} values must be positive, got {item!r}")
            raise ValueError(f"{name} values must be at least {minimum}, got {item!r}")
        result[size] = value
    return result


def _phase_args(args: argparse.Namespace) -> list[str]:
    """Return phase-diagram options shared by every generated command."""

    options: list[str] = []
    values = (
        ("--lr_min", args.lr_min),
        ("--lr_max", args.lr_max),
        ("--n_lr", args.n_lr),
        ("--wd_max", args.wd_max),
        ("--n_wd", args.n_wd),
        ("-m", args.model_type),
        ("--m_nlayer", args.m_nlayer),
        ("--m_n_heads", args.m_n_heads),
        ("--m_d_model", args.m_d_model),
        ("--m_context_len", args.m_context_len),
        ("--m_pos_encoding", args.m_pos_encoding),
        ("--random_seed", args.random_seed),
    )
    for flag, value in values:
        if value is not None:
            options.extend([flag, str(value)])
    # These are useful metadata when a non-50/50 master was prepared.  They
    # are accepted by the phase-diagram interface even when -dpath is used.
    options.extend(["-tp", str(args.train_pct), "-st", str(args.split_type)])
    if args.operand_length is not None:
        options.extend(["-ol", str(args.operand_length)])
    if args.enable_ineffective_training_stop:
        options.append("--enable_ineffective_training_stop")
    if args.enable_skip_larger_wd_after_confusion:
        options.append("--enable_skip_larger_wd_after_confusion")
    if args.enable_training_loss_plateau_stop:
        options.extend(
            [
                "--enable_training_loss_plateau_stop",
                "--training_loss_plateau_window_ratio",
                str(args.training_loss_plateau_window_ratio),
                "--training_loss_plateau_consecutive_windows",
                str(args.training_loss_plateau_consecutive_windows),
                "--training_loss_plateau_min_epoch",
                str(args.training_loss_plateau_min_epoch),
                "--training_loss_plateau_min_relative_improvement",
                str(args.training_loss_plateau_min_relative_improvement),
            ]
        )
    else:
        options.append("--disable_training_loss_plateau_stop")
    options.extend(args.phase_extra_arg)
    return options


def build_command(
    phase_script: Path,
    dataset_dir: Path,
    result_dir: Path,
    plan: DatasetPlan,
    *,
    phase_options: Sequence[str],
) -> list[str]:
    # ``--modulus`` is the operand-domain size used by the phase-diagram
    # interface.  The tokenizer copied from the master retains labels modulo
    # fixed_label_modulus.
    return [
        sys.executable,
        str(phase_script),
        "-dpath",
        str(dataset_dir),
        "--modulus",
        str(plan.size),
        "-o",
        str(result_dir),
        "-bs",
        str(plan.batchsize),
        "-epoch",
        str(plan.epoch),
        *phase_options,
    ]


def write_run_python(
    path: Path,
    experiments: dict[int, dict[str, object]],
    phase_script: Path,
    phase_options: Sequence[str],
) -> None:
    """Write an editable runner with one map entry per experiment."""

    dataset_paths = {size: config["dataset_path"] for size, config in experiments.items()}
    result_paths = {size: config["result_path"] for size, config in experiments.items()}
    batch_sizes = {size: config["batchsize"] for size, config in experiments.items()}
    epochs = {size: config["epoch"] for size, config in experiments.items()}
    sizes = list(experiments)
    text = f'''#!/usr/bin/env python3
"""Run all generated grokking phase-diagram experiments.

Edit the maps below to change one experiment without regenerating datasets.
There is intentionally one entry per size on each map.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

PHASE_SCRIPT = {str(phase_script)!r}
PHASE_OPTIONS = {list(phase_options)!r}
EXPERIMENT_SIZES = {sizes!r}

# Editable per-experiment configuration.  Keep the same sizes in each map.
DATASET_PATH_BY_SIZE = {dataset_paths!r}
RESULT_PATH_BY_SIZE = {result_paths!r}
BATCH_SIZE_BY_SIZE = {batch_sizes!r}
EPOCH_BY_SIZE = {epochs!r}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    # Keep the environment used to generate the commands by default. Set
    # PYTHON when moving the output directory to another machine/conda env.
    recorded_python = {sys.executable!r}
    python = os.environ.get("PYTHON", recorded_python)
    failed = 0
    for index, size in enumerate(EXPERIMENT_SIZES, 1):
        command = [
            python,
            PHASE_SCRIPT,
            "-dpath", DATASET_PATH_BY_SIZE[size],
            "--modulus", str(size),
            "-o", RESULT_PATH_BY_SIZE[size],
            "-bs", str(BATCH_SIZE_BY_SIZE[size]),
            "-epoch", str(EPOCH_BY_SIZE[size]),
            *PHASE_OPTIONS,
        ]
        print(f"[{{index}}/{{len(EXPERIMENT_SIZES)}}] {{' '.join(command)}}", flush=True)
        if args.dry_run:
            continue
        try:
            subprocess.run(command, cwd=os.path.dirname(PHASE_SCRIPT), check=True)
        except subprocess.CalledProcessError as error:
            failed += 1
            if not args.continue_on_error:
                return error.returncode or 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def write_run_shell(path: Path, run_python: Path) -> None:
    text = f'''#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
exec "${{PYTHON:-python3}}" "$SCRIPT_DIR/{run_python.name}" "$@"
'''
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def prepare_prefixes(
    master: Path,
    output_root: Path,
    sizes: Sequence[int],
    *,
    fixed_label_modulus: int,
    train_pct: float,
    split_type: str,
) -> list[dict]:
    """Create every prefix and return manifest entries."""

    train_source, val_source, tokenizer_source = validate_master(master)
    datasets_root = output_root / "datasets"
    sizes = tuple(sorted(set(sizes)))
    directories = {
        size: datasets_root / dataset_dir_name(size, fixed_label_modulus, train_pct, split_type)
        for size in sizes
    }
    train_total, train_kept = _write_prefixes_for_file(
        train_source,
        {size: directory / "train.txt" for size, directory in directories.items()},
        sizes,
    )
    val_total, val_kept = _write_prefixes_for_file(
        val_source,
        {size: directory / "val.txt" for size, directory in directories.items()},
        sizes,
    )

    entries = []
    for size in sizes:
        directory = directories[size]
        shutil.copyfile(tokenizer_source, directory / "tokenizer.txt")
        actual = train_kept[size] + val_kept[size]
        expected = size * size
        if actual != expected:
            raise ValueError(
                f"Prefix size {size} contains {actual} examples, expected {expected}. "
                "Check that the master dataset is complete and uses the expected binary format."
            )
        metadata = {
            "source_master": str(master),
            "dataset_dir": str(directory),
            "fixed_label_modulus": fixed_label_modulus,
            "operand_size": size,
            "expected_examples": expected,
            "train_examples": train_kept[size],
            "validation_examples": val_kept[size],
            "master_train_examples": train_total,
            "master_validation_examples": val_total,
            "tokenizer_source": str(tokenizer_source),
        }
        (directory / "prefix_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        entries.append(metadata)
    return entries


def add_arguments(parser: argparse.ArgumentParser) -> None:
    master_source = parser.add_mutually_exclusive_group(required=True)
    master_source.add_argument(
        "--master-path",
        type=Path,
        help="Existing complete master dataset directory containing train.txt, val.txt and tokenizer.txt.",
    )
    master_source.add_argument(
        "-dexp",
        "--dataset-exp",
        "--master-expression",
        dest="dataset_exp",
        help="Generate the master from an expression, e.g. x**y_mod_1997 (also accepts x**y).",
    )
    parser.add_argument(
        "--generate-master",
        action="store_true",
        help="Deprecated compatibility flag; -dexp already generates the master automatically.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to ./generate_grokking_phase_diagram_subdataset_YYYY-MM-DD_HH-MM-SS.",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--modulus", type=int, default=1997, help="Fixed label modulus of the master dataset")
    parser.add_argument("--train-pct", type=float, default=50.0)
    parser.add_argument("--split-type", default="random")
    parser.add_argument("--operand-length", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--base-size", type=int, default=DEFAULT_BASE_SIZE, help="Legacy compatibility value; batch-count scaling does not use it")
    parser.add_argument("--base-epoch", type=int, default=DEFAULT_BASE_EPOCH)
    parser.add_argument("--epoch-power", type=float, default=DEFAULT_EPOCH_POWER, help="Exponent in epoch = base_epoch / batch_count^power")
    parser.add_argument("--batchsize-cap", type=int, default=DEFAULT_BATCHSIZE_CAP)
    parser.add_argument("--batchsize", type=int, default=None, help="Override batch size for every generated run")
    parser.add_argument("--batchsize-for", action="append", default=[], metavar="SIZE=VALUE", help="Override batch size for one size; repeatable")
    parser.add_argument("--epoch", type=int, default=None, help="Override epoch count for every generated run")
    parser.add_argument("--epoch-for", action="append", default=[], metavar="SIZE=VALUE", help="Override epoch count for one size; repeatable")
    parser.add_argument("--phase-script", type=Path, default=DEFAULT_PHASE_SCRIPT)

    # Expose the phase-diagram generator's tunable interface.
    parser.add_argument("--lr_min", type=float, default=None)
    parser.add_argument("--lr_max", type=float, default=None)
    parser.add_argument("--n_lr", type=int, default=None)
    parser.add_argument("--wd_max", type=float, default=None)
    parser.add_argument("--n_wd", type=int, default=None)
    parser.add_argument("-m", "--model-type", "--model_type", dest="model_type", default=None)
    parser.add_argument("--m_nlayer", type=int, default=None)
    parser.add_argument("--m_n_heads", type=int, default=None)
    parser.add_argument("--m_d_model", type=int, default=None)
    parser.add_argument("--m_context_len", type=int, default=None)
    parser.add_argument("--m_pos_encoding", choices=["default", "trainable"], default=None)
    parser.set_defaults(
        enable_ineffective_training_stop=True,
        enable_skip_larger_wd_after_confusion=True,
        enable_training_loss_plateau_stop=True,
    )
    parser.add_argument(
        "--enable-ineffective-training-stop",
        "--enable_ineffective_training_stop",
        dest="enable_ineffective_training_stop",
        action="store_true",
        help="Enable ineffective/high-loss early stopping (enabled by default)",
    )
    parser.add_argument(
        "--disable-ineffective-training-stop",
        dest="enable_ineffective_training_stop",
        action="store_false",
        help="Do not add --enable_ineffective_training_stop to generated runs",
    )
    parser.add_argument(
        "--enable-skip-larger-wd-after-confusion",
        "--enable_skip_larger_wd_after_confusion",
        dest="enable_skip_larger_wd_after_confusion",
        action="store_true",
        help="Skip larger WD values after repeated confusion (enabled by default)",
    )
    parser.add_argument(
        "--disable-skip-larger-wd-after-confusion",
        dest="enable_skip_larger_wd_after_confusion",
        action="store_false",
        help="Do not add --enable_skip_larger_wd_after_confusion to generated runs",
    )
    parser.add_argument(
        "--enable-training-loss-plateau-stop",
        action="store_true",
        help="Add the training-loss plateau stop to generated phase-diagram runs (enabled by default)",
    )
    parser.add_argument(
        "--disable-training-loss-plateau-stop",
        action="store_false",
        dest="enable_training_loss_plateau_stop",
        help="Do not add training-loss plateau stopping to generated runs",
    )
    parser.add_argument("--training-loss-plateau-window-ratio", type=float, default=0.01)
    parser.add_argument("--training-loss-plateau-consecutive-windows", type=int, default=2)
    parser.add_argument("--training-loss-plateau-min-epoch", type=int, default=0)
    parser.add_argument("--training-loss-plateau-min-relative-improvement", type=float, default=0.01)
    parser.add_argument(
        "--phase-extra-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Additional already-tokenized phase-generator argument; repeat for each token",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args(argv)

    if args.generate_master and args.dataset_exp is None:
        parser.error("--generate-master is only valid together with -dexp/--dataset-exp")
    if args.modulus <= 0:
        parser.error("--modulus must be positive")
    if any(size <= 0 or size > args.modulus for size in args.sizes):
        parser.error(f"every --sizes value must be in [1, {args.modulus}]")
    if args.batchsize is not None and args.batchsize <= 0:
        parser.error("--batchsize must be positive")
    if args.epoch is not None and args.epoch < MIN_EPOCH:
        parser.error(f"--epoch must be at least {MIN_EPOCH}")
    if args.batchsize_cap <= 0:
        parser.error("--batchsize-cap must be positive")
    if args.base_size <= 0 or args.base_epoch < MIN_EPOCH:
        parser.error(f"--base-size must be positive and --base-epoch must be at least {MIN_EPOCH}")
    if args.epoch_power < 0:
        parser.error("--epoch-power must be non-negative")
    if args.training_loss_plateau_window_ratio <= 0:
        parser.error("--training-loss-plateau-window-ratio must be positive")
    if args.training_loss_plateau_consecutive_windows < 1:
        parser.error("--training-loss-plateau-consecutive-windows must be positive")
    if args.training_loss_plateau_min_epoch < 0:
        parser.error("--training-loss-plateau-min-epoch must be non-negative")
    if args.training_loss_plateau_min_relative_improvement < 0:
        parser.error("--training-loss-plateau-min-relative-improvement must be non-negative")
    try:
        batchsize_by_size = parse_size_overrides(args.batchsize_for, "--batchsize-for")
        epoch_by_size = parse_size_overrides(args.epoch_for, "--epoch-for", minimum=MIN_EPOCH)
    except ValueError as error:
        parser.error(str(error))

    if args.output_root is None:
        output_root = (
            Path.cwd()
            / f"{Path(__file__).stem}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        ).resolve()
    else:
        output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    requested_master = args.master_path.expanduser().resolve()
    # ``-dexp`` is intentionally sufficient to request generation.  The
    # explicit flag remains accepted for compatibility, but no longer permits
    # omitting the required source selector.
    should_generate_master = args.dataset_exp is not None
    generated_expression = None
    if should_generate_master:
        expression = args.dataset_exp
        generated_expression = expression
        source_master = generate_master_dataset(
            output_root,
            expression=expression,
            modulus=args.modulus,
            train_pct=args.train_pct,
            split_type=args.split_type,
            operand_length=args.operand_length,
            random_seed=args.random_seed,
        )
    else:
        source_master = requested_master
    validate_master(source_master)
    master = copy_master(source_master, output_root / "master")

    entries = prepare_prefixes(
        master,
        output_root,
        args.sizes,
        fixed_label_modulus=args.modulus,
        train_pct=args.train_pct,
        split_type=args.split_type,
    )
    phase_script = args.phase_script.expanduser().resolve()
    if not phase_script.is_file():
        parser.error(f"phase script does not exist: {phase_script}")
    phase_options = _phase_args(args)
    experiments: dict[int, dict[str, object]] = {}
    plans = []
    for entry in entries:
        plan = make_plan(
            entry["operand_size"],
            batchsize_cap=args.batchsize_cap,
            base_size=args.base_size,
            base_epoch=args.base_epoch,
            epoch_power=args.epoch_power,
            batchsize_override=batchsize_by_size.get(entry["operand_size"], args.batchsize),
            epoch_override=epoch_by_size.get(entry["operand_size"], args.epoch),
            train_examples=entry["train_examples"],
        )
        result_dir = output_root / "results" / result_dir_name(
            plan.size, args.modulus, args.train_pct, args.split_type
        )
        entry.update(
            {
                "result_dir": str(result_dir),
                "dataset_size": plan.dataset_size,
                "training_examples": entry["train_examples"],
                "batchsize": plan.batchsize,
                "batch_count": plan.batch_count,
                "epoch": plan.epoch,
            }
        )
        experiments[plan.size] = {
            "dataset_path": entry["dataset_dir"],
            "result_path": str(result_dir),
            "batchsize": plan.batchsize,
            "epoch": plan.epoch,
        }
        plans.append(plan)
        print(
            f"Prepared n={plan.size}: train={entry['train_examples']}, "
            f"val={entry['validation_examples']}, bs={plan.batchsize}, "
            f"batches={plan.batch_count}, epoch={plan.epoch}",
            flush=True,
        )

    run_python = output_root / "run_all.py"
    run_shell = output_root / "run_all.sh"
    write_run_python(run_python, experiments, phase_script, phase_options)
    write_run_shell(run_shell, run_python)

    manifest = {
        "master_path": str(master),
        "source_master_path": str(source_master),
        "master_expression": generated_expression,
        "output_root": str(output_root),
        "fixed_label_modulus": args.modulus,
        "sizes": [plan.size for plan in plans],
        "scaling": {
            "base_size": args.base_size,
            "base_epoch": args.base_epoch,
            "epoch_power": args.epoch_power,
            "min_epoch": MIN_EPOCH,
            "batch_count_basis": "training_examples / batchsize",
            "batchsize_cap": args.batchsize_cap,
            "batchsize_override": args.batchsize,
            "epoch_override": args.epoch,
            "batchsize_by_size": batchsize_by_size,
            "epoch_by_size": epoch_by_size,
        },
        "phase_script": str(phase_script),
        "phase_options": phase_options,
        "prefixes": entries,
        "run_python": str(run_python),
        "run_shell": str(run_shell),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote manifest: {output_root / 'manifest.json'}")
    print(f"Wrote runner: {run_shell}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
