import argparse
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.engine import Device
from py_src.ml_setup_dataset import DatasetSetup, DatasetType
from py_src.model_opti_save_load import load_model_state_file
from py_src.util import set_seed, setup_logging

if TYPE_CHECKING:
    from py_src.ml_setup import MLSetup

logger = logging.getLogger("difussion_model_sample")


class _EmptyDataset(Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, index):
        raise IndexError(index)


def _make_empty_dataset_setup(dataset_type_name: str) -> DatasetSetup:
    dataset_type = DatasetType[dataset_type_name]
    empty_dataset = _EmptyDataset()
    return DatasetSetup(dataset_type, empty_dataset, empty_dataset)


def _build_sampling_ml_setup(model_type: str, dataset_type: str, preset: int):
    from py_src.ml_setup import get_ml_setup_from_config

    if model_type == "ddpm_cifar10":
        expected_dataset = "cifar10"
        if dataset_type not in ("default", expected_dataset):
            raise RuntimeError(
                f"ddpm_cifar10 expects dataset {expected_dataset!r}, got {dataset_type!r}"
            )
        from py_src.ml_setup.ddpm_cifar import ddpm_cifar10

        return ddpm_cifar10(
            override_dataset=_make_empty_dataset_setup(expected_dataset)
        )

    if model_type == "ddpm_flowers102":
        expected_dataset = "flowers102"
        if dataset_type not in ("default", expected_dataset):
            raise RuntimeError(
                f"ddpm_flowers102 expects dataset {expected_dataset!r}, got {dataset_type!r}"
            )
        from py_src.ml_setup.ddpm_flowers import ddpm_flowers102

        return ddpm_flowers102(
            override_dataset=_make_empty_dataset_setup(expected_dataset)
        )

    return get_ml_setup_from_config(model_type, dataset_type=dataset_type, preset=preset)


def _diffusion_sampler_accepts_seed(sample_fn) -> bool:
    signature = inspect.signature(sample_fn)
    parameters = list(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return True
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return True
    if "seed" in signature.parameters:
        return True
    positional_params = [
        parameter
        for parameter in parameters
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    return len(positional_params) >= 6


def _call_diffusion_sampler(
    sample_fn,
    model: torch.nn.Module,
    output_folder: str,
    sample_tag: int,
    device: torch.device,
    count: int,
    seed: int,
) -> None:
    if _diffusion_sampler_accepts_seed(sample_fn):
        sample_fn(model, output_folder, sample_tag, device, count, seed)
        return

    set_seed(seed)
    sample_fn(model, output_folder, sample_tag, device, count)


def _resolve_model_and_dataset_type(
    args,
) -> tuple[Optional[dict], str, str]:
    model_state = None
    checkpoint_model_type = None
    checkpoint_dataset_type = None

    if args.model_path is not None:
        model_state, checkpoint_model_type, checkpoint_dataset_type = load_model_state_file(args.model_path)

    model_type = args.model_type or checkpoint_model_type
    if model_type is None:
        raise RuntimeError("model type is required when the checkpoint metadata does not provide one")

    if (
        args.model_type is not None
        and checkpoint_model_type is not None
        and args.model_type != checkpoint_model_type
    ):
        raise RuntimeError(
            f"model type mismatch: arg={args.model_type!r} checkpoint={checkpoint_model_type!r}"
        )

    dataset_type = args.dataset_type
    if dataset_type == "default" and checkpoint_dataset_type is not None:
        dataset_type = checkpoint_dataset_type

    if (
        args.dataset_type != "default"
        and checkpoint_dataset_type is not None
        and args.dataset_type != checkpoint_dataset_type
    ):
        raise RuntimeError(
            f"dataset type mismatch: arg={args.dataset_type!r} checkpoint={checkpoint_dataset_type!r}"
        )

    return model_state, model_type, dataset_type


def _default_output_folder(args, model_type: str) -> str:
    if args.output_folder is not None:
        return args.output_folder

    if args.model_path is not None:
        base_name = Path(args.model_path).stem
    else:
        base_name = model_type
    return os.path.join(os.getcwd(), f"{base_name}_samples")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate diffusion model samples with a fixed random seed"
    )
    parser.add_argument("model_path", nargs="?", default=None, help="checkpoint path (.model.pt)")
    parser.add_argument("-m", "--model_type", type=str, default=None, help="model type override")
    parser.add_argument(
        "-d",
        "--dataset_type",
        type=str,
        default="default",
        help="dataset type override",
    )
    parser.add_argument("-P", "--preset", type=int, default=0, help="factory preset index")
    parser.add_argument("-n", "--count", type=int, default=10, help="number of samples to generate")
    parser.add_argument("-s", "--seed", type=int, default=0, help="sampling random seed")
    parser.add_argument(
        "-t",
        "--sample_tag",
        type=int,
        default=None,
        help="integer tag used in output file names; defaults to the seed",
    )
    parser.add_argument("-o", "--output_folder", type=str, default=None, help="output folder")
    parser.add_argument("--cpu", action="store_true", help="force CPU sampling")
    args = parser.parse_args()

    setup_logging(logger, "main")
    logger.info("logging ready")

    if args.count < 1:
        raise RuntimeError(f"count must be positive, got {args.count}")

    if args.model_path is None and args.model_type is None:
        raise RuntimeError("either model_path or --model_type must be provided")

    model_state, model_type, dataset_type = _resolve_model_and_dataset_type(args)
    output_folder = _default_output_folder(args, model_type)
    sample_tag = args.seed if args.sample_tag is None else args.sample_tag

    if model_state is None:
        set_seed(args.seed)
        logger.info("seeded model initialization with %d because no checkpoint was provided", args.seed)

    ml_setup = _build_sampling_ml_setup(model_type, dataset_type, args.preset)
    from py_src.ml_setup import ApplicationType

    if ml_setup.application_type != ApplicationType.diffusion:
        raise RuntimeError(f"{model_type} is not a diffusion ML setup")
    if ml_setup.difussion_generate_sample is None:
        raise RuntimeError(f"{model_type} does not define difussion_generate_sample")

    if model_state is not None:
        ml_setup.model.load_state_dict(model_state)
        logger.info("loaded model state from %s", args.model_path)

    device = Device.cpu() if args.cpu else Device.auto()
    ml_setup.model.to(device.device)

    os.makedirs(output_folder, exist_ok=True)
    logger.info(
        "sampling model=%s dataset=%s count=%d seed=%d device=%s output=%s",
        model_type,
        dataset_type,
        args.count,
        args.seed,
        device.device,
        output_folder,
    )

    with torch.no_grad():
        _call_diffusion_sampler(
            ml_setup.difussion_generate_sample,
            ml_setup.model,
            output_folder,
            sample_tag,
            device.device,
            args.count,
            args.seed,
        )

    logger.info("saved %d sample(s)", args.count)


if __name__ == "__main__":
    main()
