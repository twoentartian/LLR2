import argparse
import copy
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src import ml_setup, util
from py_src.adapters import clone_adapter_for_model
from py_src.engine import Device, train as engine_train
from py_src.ml_setup_dataset.dataset_random import (
    dataset_type_to_random,
    get_random_dataset_setup,
    save_random_images,
)
from py_src.ml_setup_dataset.dataset_types import DatasetType
from py_src.ml_setup_model import ModelType
from py_src.model_opti_save_load import save_model_state


LOGGER = logging.getLogger("measure_model_capacity_of_random_data")


def _save_lr_wd(model: torch.nn.Module, optimizer: torch.optim.Optimizer, file_path: Path) -> None:
    param_to_name = {param: name for name, param in model.named_parameters()}
    decays: dict[str, float] = {}
    for param_group in optimizer.param_groups:
        weight_decay = float(param_group.get("weight_decay", 0.0))
        for param in param_group["params"]:
            name = param_to_name.get(param)
            if name is not None:
                decays[name] = weight_decay
    with open(file_path, "w", encoding="utf-8") as outfile:
        json.dump(decays, outfile, indent=4)


def _get_random_dataset_training_setup(
    current_ml_setup,
    model: torch.nn.Module,
    *,
    override_dataset,
    override_batch_size: int,
    override_epoch: Optional[int],
    override_weight_decay: Optional[float],
) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler], int]:
    model_type = current_ml_setup.model_type
    dataset_type = current_ml_setup.dataset_type
    steps_per_epoch = len(override_dataset) // override_batch_size + 1

    epochs = override_epoch
    weight_decay = override_weight_decay

    if model_type in (ModelType.lenet5, ModelType.lenet4):
        if dataset_type != DatasetType.mnist:
            raise NotImplementedError(f"Random-data preset is not implemented for {model_type.name} @ {dataset_type.name}")
        learning_rate = 0.01
        epochs = 100 if epochs is None else epochs
        weight_decay = 2e-4 if weight_decay is None else weight_decay
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        return optimizer, None, epochs

    if model_type in (ModelType.resnet18_bn, ModelType.resnet18_gn):
        learning_rate = 0.1
        epochs = 100 if epochs is None else epochs
        if dataset_type in (DatasetType.cifar10, DatasetType.cifar100):
            weight_decay = 5e-4 if weight_decay is None else weight_decay
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                learning_rate,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
            )
            return optimizer, scheduler, epochs
        if dataset_type in (DatasetType.imagenet100, DatasetType.imagenet1k):
            weight_decay = 1e-4 if weight_decay is None else weight_decay
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs * steps_per_epoch,
                eta_min=1e-3,
            )
            return optimizer, scheduler, epochs
        raise NotImplementedError(f"Random-data preset is not implemented for {model_type.name} @ {dataset_type.name}")

    if model_type == ModelType.cct_7_3x1_32:
        if dataset_type not in (DatasetType.cifar10, DatasetType.cifar100):
            raise NotImplementedError(f"Random-data preset is not implemented for {model_type.name} @ {dataset_type.name}")
        weight_decay = 6e-2 if weight_decay is None else weight_decay
        initial_lr = 55e-5
        warmup_lr = 1e-5
        min_lr = 1e-5
        warmup_epochs = 10
        epochs = 300 if epochs is None else epochs
        warmup_steps = warmup_epochs * steps_per_epoch
        cosine_steps = max((epochs - warmup_epochs) * steps_per_epoch, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps and warmup_steps > 0:
                learning_rate = warmup_lr + (initial_lr - warmup_lr) * (current_step / warmup_steps)
            elif current_step < warmup_steps + cosine_steps:
                progress = current_step - warmup_steps
                learning_rate = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress / cosine_steps))
            else:
                learning_rate = min_lr
            return learning_rate / initial_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, scheduler, epochs

    raise NotImplementedError(f"Random-data preset is not implemented for {model_type.name} @ {dataset_type.name}")


def _build_training_dataloader(training_data, batch_size: int, core_count: int, device: Device) -> DataLoader:
    num_workers = min(max(core_count, 0), 8)
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "pin_memory": device.device.type == "cuda",
        "num_workers": num_workers,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 4
    return DataLoader(training_data, **dataloader_kwargs)


def _check_number_of_sample(
    sample_count_per_label: int,
    *,
    random_dataset_type: DatasetType,
    output_dir: Path,
    current_ml_setup,
    accuracy_threshold: float,
    device: Device,
    use_amp: bool,
    core_count: int,
    dataset_gen_worker: Optional[int],
    dataset_gen_reset_seed_per_label: bool,
    dataset_gen_reset_seed_per_sample: bool,
    override_epoch: Optional[int],
    override_weight_decay: Optional[float],
) -> tuple[float, float]:
    dataset_path = output_dir / f"random_dataset_count_{sample_count_per_label}"
    save_random_images(
        sample_count_per_label,
        random_dataset_type,
        str(dataset_path),
        num_workers=dataset_gen_worker,
        reset_random_seeds_per_label=dataset_gen_reset_seed_per_label,
        reset_random_seeds_per_sample=dataset_gen_reset_seed_per_sample,
    )

    train_path = output_dir / f"train_on_random_dataset_count_{sample_count_per_label}"
    train_path.mkdir(parents=True, exist_ok=True)

    dataset_setup = get_random_dataset_setup(random_dataset_type, override_dataset_path=str(dataset_path))
    training_data = dataset_setup.train_data
    default_batch_size = current_ml_setup.default_batch_size if current_ml_setup.default_batch_size > 0 else len(training_data) # type: ignore
    batch_size = min(len(training_data), default_batch_size) # type: ignore

    model = copy.deepcopy(current_ml_setup.model)
    util.re_initialize_model(model)
    model.to(device.device)
    adapter = clone_adapter_for_model(current_ml_setup.adapter, model, criterion=current_ml_setup.criterion)

    optimizer, lr_scheduler, epochs = _get_random_dataset_training_setup(
        current_ml_setup,
        model,
        override_dataset=training_data,
        override_batch_size=batch_size,
        override_epoch=override_epoch,
        override_weight_decay=override_weight_decay,
    )
    _save_lr_wd(model, optimizer, train_path / "lr_wd.txt")

    dataloader = _build_training_dataloader(training_data, batch_size, core_count, device)
    scaler = device.make_scaler() if use_amp and device.supports_amp else None

    log_file_path = train_path / "train.log"
    with open(log_file_path, "w", encoding="utf-8") as logfile:
        logfile.write("epoch,loss,accuracy,lrs\n")
        LOGGER.info("begin training (count: %s)", sample_count_per_label)

        final_loss = 0.0
        final_accuracy = 0.0
        for epoch in range(epochs):
            train_result = engine_train(
                adapter,
                dataloader,
                optimizer,
                lr_scheduler,
                device=device,
                scaler=scaler,
            )
            lrs = [param_group["lr"] for param_group in optimizer.param_groups]
            final_loss = train_result.avg_loss
            final_accuracy = float(train_result.accuracy or 0.0)

            save_model_state(
                str(train_path / f"{epoch}.model.pt"),
                model.state_dict(),
                current_ml_setup.model_type.name,
                dataset_setup.dataset_type.name,
            )
            LOGGER.info("epoch[%s] loss=%s accuracy=%s lrs=%s", epoch, final_loss, final_accuracy, lrs)
            logfile.write(f"{epoch},{final_loss},{final_accuracy},{lrs}\n")
            logfile.flush()

            if final_accuracy > accuracy_threshold:
                LOGGER.info(
                    "early stopping at epoch %s, %s(accuracy) > %s(threshold)",
                    epoch,
                    final_accuracy,
                    accuracy_threshold,
                )
                break

    LOGGER.info("finish training")
    return final_loss, final_accuracy


def _measure_one_configuration(
    *,
    random_dataset_type: DatasetType,
    output_folder_path: Path,
    current_ml_setup,
    accuracy_threshold: float,
    device: Device,
    use_amp: bool,
    core_count: int,
    dataset_gen_worker: Optional[int],
    dataset_gen_reset_seed_per_label: bool,
    dataset_gen_reset_seed_per_sample: bool,
    override_epoch: Optional[int],
    override_weight_decay: Optional[float],
) -> None:
    low = 1
    LOGGER.info("try sample_count per label: %s.", low)
    _, accuracy = _check_number_of_sample(
        low,
        random_dataset_type=random_dataset_type,
        output_dir=output_folder_path,
        current_ml_setup=current_ml_setup,
        accuracy_threshold=accuracy_threshold,
        device=device,
        use_amp=use_amp,
        core_count=core_count,
        dataset_gen_worker=dataset_gen_worker,
        dataset_gen_reset_seed_per_label=dataset_gen_reset_seed_per_label,
        dataset_gen_reset_seed_per_sample=dataset_gen_reset_seed_per_sample,
        override_epoch=override_epoch,
        override_weight_decay=override_weight_decay,
    )
    if accuracy < accuracy_threshold:
        LOGGER.critical("The accuracy of random_dataset_count_%s is smaller than %s. Stopped.", low, accuracy_threshold)

    high = 2
    LOGGER.info("try sample_count per label: %s.", high)
    _, accuracy = _check_number_of_sample(
        high,
        random_dataset_type=random_dataset_type,
        output_dir=output_folder_path,
        current_ml_setup=current_ml_setup,
        accuracy_threshold=accuracy_threshold,
        device=device,
        use_amp=use_amp,
        core_count=core_count,
        dataset_gen_worker=dataset_gen_worker,
        dataset_gen_reset_seed_per_label=dataset_gen_reset_seed_per_label,
        dataset_gen_reset_seed_per_sample=dataset_gen_reset_seed_per_sample,
        override_epoch=override_epoch,
        override_weight_decay=override_weight_decay,
    )
    while accuracy >= accuracy_threshold:
        low = high
        high *= 2
        LOGGER.info("try sample_count per label: %s.", high)
        _, accuracy = _check_number_of_sample(
            high,
            random_dataset_type=random_dataset_type,
            output_dir=output_folder_path,
            current_ml_setup=current_ml_setup,
            accuracy_threshold=accuracy_threshold,
            device=device,
            use_amp=use_amp,
            core_count=core_count,
            dataset_gen_worker=dataset_gen_worker,
            dataset_gen_reset_seed_per_label=dataset_gen_reset_seed_per_label,
            dataset_gen_reset_seed_per_sample=dataset_gen_reset_seed_per_sample,
            override_epoch=override_epoch,
            override_weight_decay=override_weight_decay,
        )

    while True:
        mid = (low + high) // 2
        if mid == low or mid == high:
            LOGGER.info("the maximum sample count is %s.", mid)
            return
        LOGGER.info("try sample_count per label: %s.", mid)
        _, accuracy = _check_number_of_sample(
            mid,
            random_dataset_type=random_dataset_type,
            output_dir=output_folder_path,
            current_ml_setup=current_ml_setup,
            accuracy_threshold=accuracy_threshold,
            device=device,
            use_amp=use_amp,
            core_count=core_count,
            dataset_gen_worker=dataset_gen_worker,
            dataset_gen_reset_seed_per_label=dataset_gen_reset_seed_per_label,
            dataset_gen_reset_seed_per_sample=dataset_gen_reset_seed_per_sample,
            override_epoch=override_epoch,
            override_weight_decay=override_weight_decay,
        )

        if accuracy < accuracy_threshold:
            high = mid
        else:
            low = mid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure model capacity in terms of how many random samples can be memorized",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="specify the model type")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="specify the dataset type")
    parser.add_argument("-o", "--output_folder_name", default=None, help="specify the output folder name")
    parser.add_argument("-c", "--core", type=int, default=os.cpu_count() or 1, help="specify the number of CPU cores to use")
    parser.add_argument("--amp", action="store_true", help="enable auto mixed precision")
    parser.add_argument("-t", "--accuracy_threshold", type=float, default=0.5, help="specify the accuracy threshold")
    parser.add_argument("--dataset_gen_worker", type=int, default=None, help="enable multiprocessing during dataset generation")
    parser.add_argument("--dataset_gen_reset_seed_per_label", action="store_true", help="reset the random seed after generating for each label")
    parser.add_argument("--dataset_gen_reset_seed_per_sample", action="store_true", help="reset the random seed after generating for each sample")
    parser.add_argument("-e", "--epoch", type=int, default=None, help="specify the number of epochs, None=default")
    parser.add_argument("-w", "--weight_decay", nargs="+", type=float, default=None, help="specify the weight decay, can be a list")
    parser.add_argument("-s", "--sample_size", type=int, default=None, help="only measure whether the model can memorize this sample size")

    args = parser.parse_args()

    time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    if args.output_folder_name is None:
        output_folder_path = Path.cwd() / f"{Path(__file__).stem}_{time_now_str}"
    else:
        output_folder_path = Path.cwd() / args.output_folder_name
    output_folder_path.mkdir(parents=True, exist_ok=True)

    util.setup_logging(LOGGER, "main", log_file_path=output_folder_path / "log.txt", exit_on_critical=True)

    current_ml_setup = ml_setup.get_ml_setup_from_config(args.model, dataset_type=args.dataset)
    random_dataset_type = dataset_type_to_random(current_ml_setup.dataset_type)
    LOGGER.info("Random dataset type: %s", random_dataset_type.name)

    device = Device.auto()
    if args.amp and not device.supports_amp:
        LOGGER.warning("AMP requested but %s does not support AMP; continuing without AMP", device.device)

    if args.sample_size is not None:
        LOGGER.info("sample size is %s", args.sample_size)
        if isinstance(args.weight_decay, list):
            LOGGER.info("mode: measure multiple weights decay for single sample size")
            LOGGER.info("wd is a list: %s", args.weight_decay)
            for weight_decay in args.weight_decay:
                output_folder_path_wd = output_folder_path / f"wd_{weight_decay}"
                output_folder_path_wd.mkdir(parents=True, exist_ok=True)
                _check_number_of_sample(
                    args.sample_size,
                    random_dataset_type=random_dataset_type,
                    output_dir=output_folder_path_wd,
                    current_ml_setup=current_ml_setup,
                    accuracy_threshold=args.accuracy_threshold,
                    device=device,
                    use_amp=args.amp,
                    core_count=args.core,
                    dataset_gen_worker=args.dataset_gen_worker,
                    dataset_gen_reset_seed_per_label=args.dataset_gen_reset_seed_per_label,
                    dataset_gen_reset_seed_per_sample=args.dataset_gen_reset_seed_per_sample,
                    override_epoch=args.epoch,
                    override_weight_decay=weight_decay,
                )
        else:
            LOGGER.info("mode: measure single weights decay for single sample size")
            LOGGER.info("wd is a single value: %s", args.weight_decay)
            override_weight_decay = args.weight_decay[0] if isinstance(args.weight_decay, list) else args.weight_decay
            _check_number_of_sample(
                args.sample_size,
                random_dataset_type=random_dataset_type,
                output_dir=output_folder_path,
                current_ml_setup=current_ml_setup,
                accuracy_threshold=args.accuracy_threshold,
                device=device,
                use_amp=args.amp,
                core_count=args.core,
                dataset_gen_worker=args.dataset_gen_worker,
                dataset_gen_reset_seed_per_label=args.dataset_gen_reset_seed_per_label,
                dataset_gen_reset_seed_per_sample=args.dataset_gen_reset_seed_per_sample,
                override_epoch=args.epoch,
                override_weight_decay=override_weight_decay,
            )
        return

    if isinstance(args.weight_decay, list):
        LOGGER.info("measure multiple weights decay mode")
        LOGGER.info("wd is a list: %s", args.weight_decay)
        for weight_decay in args.weight_decay:
            output_folder_path_wd = output_folder_path / f"wd_{weight_decay}"
            output_folder_path_wd.mkdir(parents=True, exist_ok=True)
            _measure_one_configuration(
                random_dataset_type=random_dataset_type,
                output_folder_path=output_folder_path_wd,
                current_ml_setup=current_ml_setup,
                accuracy_threshold=args.accuracy_threshold,
                device=device,
                use_amp=args.amp,
                core_count=args.core,
                dataset_gen_worker=args.dataset_gen_worker,
                dataset_gen_reset_seed_per_label=args.dataset_gen_reset_seed_per_label,
                dataset_gen_reset_seed_per_sample=args.dataset_gen_reset_seed_per_sample,
                override_epoch=args.epoch,
                override_weight_decay=weight_decay,
            )
    else:
        LOGGER.info("measure single weights decay mode")
        LOGGER.info("wd is a single value: %s", args.weight_decay)
        override_weight_decay = args.weight_decay[0] if isinstance(args.weight_decay, list) else args.weight_decay
        _measure_one_configuration(
            random_dataset_type=random_dataset_type,
            output_folder_path=output_folder_path,
            current_ml_setup=current_ml_setup,
            accuracy_threshold=args.accuracy_threshold,
            device=device,
            use_amp=args.amp,
            core_count=args.core,
            dataset_gen_worker=args.dataset_gen_worker,
            dataset_gen_reset_seed_per_label=args.dataset_gen_reset_seed_per_label,
            dataset_gen_reset_seed_per_sample=args.dataset_gen_reset_seed_per_sample,
            override_epoch=args.epoch,
            override_weight_decay=override_weight_decay,
        )


if __name__ == "__main__":
    main()
