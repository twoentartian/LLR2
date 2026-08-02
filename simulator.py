from __future__ import annotations

import argparse
import importlib.util
import itertools
import logging
import os
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import torch
import networkx as nx

from py_src import util
from py_src.engine import Device, train as engine_train
from py_src.ml_setup import MLSetup
from py_src.ml_setup_dataset.dataset_intermediate_layer import (
    DatasetWithFastLabelSelection,
)
from py_src.node import Node, ensure_ml_setup_compatibility
from py_src.shared_dataloader import BatchRequest, SharedDataLoader
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase


LOGGER_NAME = "SimulatorBase"
LOG_FILE_NAME = "info.log"
BACKUP_DIR_NAME = "backup"
CONFIG_MODULE_NAME = "config"
REPORT_FINISH_TIME_PER_TICK = 100
DEFAULT_DEVICE = "auto"

SIMULATOR_LOGGER = logging.getLogger(LOGGER_NAME)


def load_configuration(config_file_path: str):
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"config file ({config_file_path}) does not exist")
    spec = importlib.util.spec_from_file_location(CONFIG_MODULE_NAME, config_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import config file {config_file_path}")
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[CONFIG_MODULE_NAME] = config_module
    spec.loader.exec_module(config_module)
    return config_module


def check_consistent_nodes(topology_generation_function, total_tick: int) -> set[int]:
    parameters = RuntimeParameters()
    parameters.max_tick = total_tick
    parameters.phase = SimulationPhase.INITIALIZING
    topology = topology_generation_function(parameters)
    if topology is None:
        raise ValueError("get_topology() must return a topology during initialization")

    previous_nodes = set(topology.nodes())
    initial_edge_count = len(topology.edges())
    max_edge_count = initial_edge_count

    for tick in range(total_tick + 1):
        parameters.phase = SimulationPhase.START_OF_TICK
        parameters.current_tick = tick
        topology = topology_generation_function(parameters)
        if topology is None:
            continue
        current_nodes = set(topology.nodes())
        max_edge_count = max(max_edge_count, len(topology.edges()))
        if previous_nodes != current_nodes:
            extra_nodes = current_nodes - previous_nodes
            missing_nodes = previous_nodes - current_nodes
            SIMULATOR_LOGGER.critical(
                "nodes (count:%s) at tick %s differ from previous nodes (count:%s). Extra: %s, missing: %s",
                len(current_nodes),
                tick,
                len(previous_nodes),
                extra_nodes,
                missing_nodes,
            )
            raise SystemExit(1)

    SIMULATOR_LOGGER.info(
        "total nodes: %s, initial edges: %s, max edge count: %s",
        len(previous_nodes),
        initial_edge_count,
        max_edge_count,
    )
    return previous_nodes


def save_topology_to_file(topology: nx.Graph, current_tick: int, output_path: str) -> None:
    topology_folder = os.path.join(output_path, "topology")
    os.makedirs(topology_folder, exist_ok=True)
    with open(os.path.join(topology_folder, f"{current_tick}.pickle"), "wb") as topology_file:
        pickle.dump(topology, topology_file)


def _slice_batches(loader: Iterable, batch_limit: int):
    return itertools.islice(iter(loader), max(0, int(batch_limit)))


def _build_shared_batch_plan(nodes: Iterable[Node]) -> list[BatchRequest]:
    requests = []
    for node_target in nodes:
        batch_count = max(0, int(node_target.num_of_batch_per_training))
        batch_size = int(node_target.ml_setup.training_batch_size)
        if batch_count == 0:
            continue
        if batch_size <= 0:
            raise ValueError(f"node {node_target.name} has an invalid batch size")

        indices = node_target.sample_training_indices(batch_count * batch_size)
        for batch_index in range(batch_count):
            start = batch_index * batch_size
            requests.append(
                BatchRequest.from_indices(
                    node_target.name,
                    batch_index,
                    indices[start : start + batch_size],
                )
            )
    return requests


def _take_routed_node_batches(
    routed_batch_iterator,
    node_name: int,
    batch_count: int,
):
    for expected_batch_index in range(max(0, int(batch_count))):
        try:
            routed_batch = next(routed_batch_iterator)
        except StopIteration as error:
            raise RuntimeError(
                f"shared DataLoader stopped before producing batches for node {node_name}"
            ) from error
        if (
            routed_batch.node_name != node_name
            or routed_batch.batch_index != expected_batch_index
        ):
            raise RuntimeError(
                "shared DataLoader returned a batch out of plan order: "
                f"expected node {node_name} batch {expected_batch_index}, "
                f"received node {routed_batch.node_name} "
                f"batch {routed_batch.batch_index}"
            )
        yield routed_batch.batch


def _refresh_label_distributions(runtime_parameters: RuntimeParameters, config_file) -> None:
    for single_node in runtime_parameters.node_container.values():
        new_label_distribution = config_file.get_label_distribution(single_node, runtime_parameters)
        if new_label_distribution is not None:
            single_node.set_label_distribution(new_label_distribution)
            SIMULATOR_LOGGER.info(
                "update label distribution to %s for %s.",
                new_label_distribution,
                single_node.name,
            )


def _trigger_services(runtime_parameters: RuntimeParameters) -> None:
    for service_inst in runtime_parameters.service_container.values():
        service_inst.trigger(runtime_parameters)


def simulation_phase_start_of_tick(runtime_parameters: RuntimeParameters) -> None:
    runtime_parameters.phase = SimulationPhase.START_OF_TICK
    _trigger_services(runtime_parameters)
    for node_target in runtime_parameters.node_container.values():
        node_target.reset_status_flags()
    SIMULATOR_LOGGER.info("current tick: %s/%s", runtime_parameters.current_tick, runtime_parameters.max_tick)


def simulation_phase_before_training(runtime_parameters: RuntimeParameters) -> None:
    runtime_parameters.phase = SimulationPhase.BEFORE_TRAINING
    _trigger_services(runtime_parameters)


def simulation_phase_training(runtime_parameters: RuntimeParameters, config_file) -> None:
    runtime_parameters.phase = SimulationPhase.TRAINING
    _trigger_services(runtime_parameters)

    training_node_names = []
    for node_name, node_target in runtime_parameters.node_container.items():
        if node_target.next_training_tick == runtime_parameters.current_tick:
            training_node_names.append(node_name)

    training_nodes = [
        runtime_parameters.node_container[node_name]
        for node_name in training_node_names
    ]
    shared_loader = runtime_parameters.shared_training_loader
    routed_batch_iterator = None
    if shared_loader is not None and training_nodes:
        shared_loader.set_plan(_build_shared_batch_plan(training_nodes))
        routed_batch_iterator = iter(shared_loader)

    for node_name, node_target in zip(training_node_names, training_nodes):
        if routed_batch_iterator is None:
            train_loader = node_target.get_data_loader()
            batches = _slice_batches(
                train_loader,
                node_target.num_of_batch_per_training,
            )
        else:
            batches = _take_routed_node_batches(
                routed_batch_iterator,
                node_name,
                node_target.num_of_batch_per_training,
            )
        training_batch_count = 0
        if node_target.enable_training and not getattr(runtime_parameters, "performance_disable_training", False):
            result = engine_train(
                node_target._adapter,
                batches,
                node_target.optimizer,
                node_target.lr_scheduler,
                device=node_target.device_obj,
                scaler=node_target.scaler,
                gradient_accumulate_every=getattr(node_target.ml_setup, "gradient_accumulate_every", 1),
                max_grad_norm=getattr(node_target.ml_setup, "max_grad_norm", None),
            )
            training_batch_count = result.iterations
            node_target.most_recent_loss = torch.tensor(result.avg_loss, device=node_target.device)
            node_target.most_recent_accuracy = float(result.accuracy or 0.0)
        else:
            for _ in batches:
                training_batch_count += 1

        if node_target.optimizer is not None:
            node_target.most_recent_lrs = [group.get("lr", 0.0) for group in node_target.optimizer.param_groups]
        else:
            node_target.most_recent_lrs = []

        if training_batch_count > 0 and node_target.enable_sending:
            node_target.is_training_this_tick = True

        most_recent_lrs_str = [f"{value:.3e}" for value in node_target.most_recent_lrs]
        SIMULATOR_LOGGER.info(
            "tick: %s, training node: %s for %s times, loss=%.4f, lrs=%s",
            runtime_parameters.current_tick,
            node_target.name,
            training_batch_count,
            float(node_target.most_recent_loss.detach().cpu().item()),
            most_recent_lrs_str,
        )

        for handler in getattr(node_target.ml_setup, "func_handler_post_training", []):
            handler(node_target.model)

    for node_name in training_node_names:
        node_target = runtime_parameters.node_container[node_name]
        node_target.next_training_tick = config_file.get_next_training_time(node_target, runtime_parameters)


def simulation_phase_after_training(runtime_parameters: RuntimeParameters) -> None:
    runtime_parameters.phase = SimulationPhase.AFTER_TRAINING
    _trigger_services(runtime_parameters)


def simulation_phase_before_averaging(runtime_parameters: RuntimeParameters) -> None:
    runtime_parameters.phase = SimulationPhase.BEFORE_AVERAGING
    _trigger_services(runtime_parameters)


def simulation_phase_averaging(runtime_parameters: RuntimeParameters) -> None:
    runtime_parameters.phase = SimulationPhase.AVERAGING
    _trigger_services(runtime_parameters)

    if getattr(runtime_parameters, "performance_disable_communication", False):
        return

    nodes_averaged = set()
    for node_target in runtime_parameters.node_container.values():
        if not node_target.is_training_this_tick:
            continue
        if not node_target.is_sending_model():
            continue

        model_stat = node_target.get_model_stat()
        if getattr(runtime_parameters, "average_on_cpu", False):
            model_stat = {
                key: value.detach().cpu()
                if torch.is_tensor(value)
                else value
                for key, value in model_stat.items()
            }

        for neighbor in runtime_parameters.topology.neighbors(node_target.name):
            runtime_parameters.node_container[neighbor].add_model_to_buffer(model_stat)

    for node_name in runtime_parameters.node_container:
        if runtime_parameters.node_container[node_name].check_averaging():
            nodes_averaged.add(node_name)

    if nodes_averaged:
        SIMULATOR_LOGGER.info(
            "tick: %s, averaging on %s nodes: %s",
            runtime_parameters.current_tick,
            len(nodes_averaged),
            nodes_averaged,
        )


def simulation_phase_after_averaging(runtime_parameters: RuntimeParameters) -> None:
    runtime_parameters.phase = SimulationPhase.AFTER_AVERAGING
    _trigger_services(runtime_parameters)


def simulation_phase_end_of_tick(runtime_parameters: RuntimeParameters) -> None:
    runtime_parameters.phase = SimulationPhase.END_OF_TICK
    _trigger_services(runtime_parameters)


def begin_simulation(runtime_parameters: RuntimeParameters, config_file) -> None:
    timer = datetime.now().timestamp()

    while runtime_parameters.current_tick <= config_file.max_tick:
        runtime_parameters.phase = SimulationPhase.START_OF_TICK

        new_topology = config_file.get_topology(runtime_parameters)
        if new_topology is not None:
            save_topology_to_file(new_topology, runtime_parameters.current_tick, runtime_parameters.output_path)
            runtime_parameters.topology = new_topology
            SIMULATOR_LOGGER.info("topology is updated at tick %s", runtime_parameters.current_tick)

        _refresh_label_distributions(runtime_parameters, config_file)

        if runtime_parameters.current_tick % REPORT_FINISH_TIME_PER_TICK == 0 and runtime_parameters.current_tick != 0:
            now = datetime.now().timestamp()
            time_elapsed = now - timer
            timer = now
            remaining = (config_file.max_tick - runtime_parameters.current_tick) // REPORT_FINISH_TIME_PER_TICK
            finish_time = now + remaining * time_elapsed
            SIMULATOR_LOGGER.info(
                "time taken for %s ticks: %.2fs, expected to finish at %s",
                REPORT_FINISH_TIME_PER_TICK,
                time_elapsed,
                datetime.fromtimestamp(finish_time),
            )

        simulation_phase_start_of_tick(runtime_parameters)
        config_file.node_behavior_control(runtime_parameters)

        simulation_phase_before_training(runtime_parameters)
        config_file.node_behavior_control(runtime_parameters)

        simulation_phase_training(runtime_parameters, config_file)
        config_file.node_behavior_control(runtime_parameters)

        simulation_phase_after_training(runtime_parameters)
        config_file.node_behavior_control(runtime_parameters)

        simulation_phase_before_averaging(runtime_parameters)
        config_file.node_behavior_control(runtime_parameters)

        simulation_phase_averaging(runtime_parameters)
        config_file.node_behavior_control(runtime_parameters)

        simulation_phase_after_averaging(runtime_parameters)
        config_file.node_behavior_control(runtime_parameters)

        simulation_phase_end_of_tick(runtime_parameters)
        config_file.node_behavior_control(runtime_parameters)

        runtime_parameters.current_tick += 1


def _resolve_output_folder(config_file, output_folder_name: Optional[str]) -> str:
    output_folder_path = getattr(config_file, "save_name", None)
    if output_folder_path is None:
        if output_folder_name is None:
            output_folder_path = os.path.join(os.curdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
        else:
            output_folder_path = os.path.join(os.curdir, output_folder_name)
    if os.path.exists(output_folder_path):
        raise FileExistsError(f"{output_folder_path} exists.")
    os.mkdir(output_folder_path)
    backup_path = os.path.join(output_folder_path, BACKUP_DIR_NAME)
    os.mkdir(backup_path)
    return output_folder_path


def _create_device(device_name: str, force_use_cpu: bool) -> Device:
    if force_use_cpu:
        return Device.cpu()
    if device_name == DEFAULT_DEVICE:
        return Device.auto()
    return Device(device_name)


def main(
    config_file_path: str,
    output_folder_name: Optional[str] = None,
    *,
    device_name: str = DEFAULT_DEVICE,
    enable_amp: bool = False,
) -> None:
    config_file = load_configuration(config_file_path)
    output_folder_path = _resolve_output_folder(config_file, output_folder_name)
    backup_path = os.path.join(output_folder_path, BACKUP_DIR_NAME)

    util.setup_logging(
        SIMULATOR_LOGGER,
        LOGGER_NAME,
        log_file_path=os.path.join(output_folder_path, LOG_FILE_NAME),
        exit_on_critical=True,
    )

    shutil.copy2(config_file_path, backup_path)
    SIMULATOR_LOGGER.info(
        "config file path: (%s), name: (%s).",
        config_file_path,
        getattr(config_file, "config_name", Path(config_file_path).name),
    )

    config_ml_setup = ensure_ml_setup_compatibility(config_file.get_ml_setup())

    runtime_parameters = RuntimeParameters()
    runtime_parameters.max_tick = config_file.max_tick
    runtime_parameters.current_tick = 0
    runtime_parameters.dataset_label = config_ml_setup.dataset_label
    runtime_parameters.phase = SimulationPhase.INITIALIZING
    runtime_parameters.output_path = output_folder_path
    runtime_parameters.mpi_enabled = False
    runtime_parameters.performance_disable_training = getattr(config_file, "performance_disable_training", False)
    runtime_parameters.performance_disable_communication = getattr(config_file, "performance_disable_communication", False)
    runtime_parameters.average_on_cpu = False

    nodes_set = check_consistent_nodes(config_file.get_topology, config_file.max_tick)
    topology = config_file.get_topology(runtime_parameters)
    runtime_parameters.topology = topology
    save_topology_to_file(topology, runtime_parameters.current_tick, runtime_parameters.output_path)
    SIMULATOR_LOGGER.info("topology is updated at tick %s", runtime_parameters.current_tick)

    device = _create_device(device_name, getattr(config_file, "force_use_cpu", False))
    SIMULATOR_LOGGER.info("simulation device: %s", device.device)
    SIMULATOR_LOGGER.info("automatic mixed precision: %s", enable_amp and device.device.type == "cuda")

    training_dataset = DatasetWithFastLabelSelection(config_ml_setup.training_data, config_ml_setup)

    shared_worker_count = getattr(
        config_file,
        "preset_shared_training_loader_workers",
        0,
    )
    shared_prefetch_factor = getattr(
        config_file,
        "preset_shared_training_loader_prefetch_factor",
        2,
    )
    if training_dataset.supports_shared_loading:
        runtime_parameters.shared_training_loader = SharedDataLoader(
            training_dataset.raw_dataset,
            num_workers=shared_worker_count,
            collate_fn=config_ml_setup.collate_fn,
            pin_memory=True,
            prefetch_factor=shared_prefetch_factor,
        )
        SIMULATOR_LOGGER.info(
            "shared training DataLoader workers: %s, prefetch factor: %s",
            shared_worker_count,
            shared_prefetch_factor,
        )
    else:
        runtime_parameters.shared_training_loader = None
        SIMULATOR_LOGGER.info(
            "training dataset requires its dedicated loader backend; "
            "shared DataLoader is disabled"
        )

    runtime_parameters.node_container = {}
    for single_node in sorted(nodes_set):
        temp_node = Node(
            single_node,
            config_ml_setup,
            device=device,
            use_cpu=getattr(config_file, "force_use_cpu", False),
            enable_amp=enable_amp,
        )
        optimizer, lr_scheduler = config_file.get_optimizer(temp_node, temp_node.model, runtime_parameters, config_ml_setup)
        temp_node.set_optimizer(optimizer)
        if lr_scheduler is not None:
            temp_node.set_lr_scheduler(lr_scheduler)

        temp_node.set_ml_setup(config_ml_setup)
        temp_node.set_next_training_tick(config_file.get_next_training_time(temp_node, runtime_parameters))
        temp_node.set_average_algorithm(config_file.get_average_algorithm(temp_node, runtime_parameters))
        temp_node.set_average_buffer_size(config_file.get_average_buffer_size(temp_node, runtime_parameters))

        label_distribution = config_file.get_label_distribution(temp_node, runtime_parameters)
        temp_node.set_label_distribution(
            label_distribution,
            dataset_with_fast_label=training_dataset,
        )

        runtime_parameters.node_container[single_node] = temp_node

    config_file.node_behavior_control(runtime_parameters)

    service_list = config_file.get_service_list()
    for service_inst in service_list:
        service_inst.initialize(
            runtime_parameters,
            output_folder_path,
            config_file=config_file,
            ml_setup=config_ml_setup,
            device=device,
            cuda_env=None,
            gpu=None,
        )
        runtime_parameters.service_container[service_inst.get_service_name()] = service_inst

    if hasattr(config_file, "preset_averaging_on_cpu"):
        if config_file.preset_averaging_on_cpu is None:
            runtime_parameters.average_on_cpu = True
        else:
            runtime_parameters.average_on_cpu = config_file.preset_averaging_on_cpu

    try:
        begin_simulation(runtime_parameters, config_file)
    finally:
        if runtime_parameters.shared_training_loader is not None:
            runtime_parameters.shared_training_loader.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="DFL simulator ported to LLR2")
    parser.add_argument("--config", type=str, default="./simulator_config.py", help='path to config file, default: "./simulator_config.py"')
    parser.add_argument("-o", "--output_folder_name", default=None, help="specify the output folder name")
    parser.add_argument("-T", "--thread", default=1, type=int, help="specify the number of thread for pytorch")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help='device to use: "auto", "cpu", "cuda", or "cuda:0"')
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision on CUDA")
    args = parser.parse_args()

    torch.set_num_threads(args.thread)
    main(args.config, args.output_folder_name, device_name=args.device, enable_amp=args.amp)
