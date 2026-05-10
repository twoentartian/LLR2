#!/usr/bin/env python3
"""Export torchvision/CCT pretrained checkpoints into the LLR2 .model.pt format."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model.pretrained_models import (
    create_torchvision_model,
    get_supported_torchvision_pretrained_model_types,
    get_torchvision_pretrained_export_specs,
)
from py_src.model_opti_save_load import save_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch pretrained models to LLR2 .model.pt files")
    parser.add_argument("-o", "--output-folder-name", "--output_folder_name", default="pytorch_pretrained", help="output folder for exported checkpoints")
    parser.add_argument("--skip-existing", "--skip_existing", action="store_true", help="skip models whose output files already exist")
    parser.add_argument("--model-types", nargs="*", default=None, help="optional subset of model types to export")
    parser.add_argument("--variants", nargs="*", default=None, help="optional subset of pretrained variants to export, for example imagenet1k_v1 imagenet1k_v2")
    parser.add_argument("--list", action="store_true", help="print the available export specs and exit")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.model_types is None:
        return
    supported = set(get_supported_torchvision_pretrained_model_types())
    unknown = sorted(set(args.model_types) - supported)
    if unknown:
        raise ValueError(f"unsupported model types: {', '.join(unknown)}")


def iter_filtered_specs(args: argparse.Namespace):
    selected_model_types = set(args.model_types) if args.model_types else None
    selected_variants = set(args.variants) if args.variants else None
    for spec in get_torchvision_pretrained_export_specs():
        if selected_model_types is not None and spec.model_type not in selected_model_types:
            continue
        if selected_variants is not None and spec.variant not in selected_variants:
            continue
        yield spec


def main() -> None:
    args = parse_args()
    validate_args(args)

    export_specs = list(iter_filtered_specs(args))
    if not export_specs:
        raise ValueError("no pretrained export specs matched the requested filters")
    if args.list:
        for spec in export_specs:
            print(f"{spec.file_name}\tmodel_type={spec.model_type}\tvariant={spec.variant}")
        return

    output_dir = Path(args.output_folder_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in export_specs:
        target_path = output_dir / spec.file_name
        if args.skip_existing and target_path.exists():
            print(f"[info] skip existing {target_path}")
            continue

        print(f"[info] exporting {spec.file_name} ({spec.model_type}, {spec.variant})")
        model = create_torchvision_model(spec.model_type, variant=spec.variant)
        save_model_state(str(target_path), model.state_dict(), spec.model_type, DatasetType.imagenet1k.name)
        print(f"[info] saved {target_path}")


if __name__ == "__main__":
    main()
