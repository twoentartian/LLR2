"""Simulation phases and shared runtime state for the LLR2 simulator."""

from enum import Enum
from typing import Any


class SimulationPhase(Enum):
    START_OF_TICK = 0
    END_OF_TICK = 7
    INITIALIZING = 8

    BEFORE_TRAINING = 1
    TRAINING = 2
    AFTER_TRAINING = 3

    BEFORE_AVERAGING = 4
    AVERAGING = 5
    AFTER_AVERAGING = 6


class RuntimeParameters:
    max_tick: int
    current_tick: int
    node_container: dict[int, Any]
    dataset_label: list[Any]
    phase: SimulationPhase
    topology: Any
    service_container: dict[str, Any]
    mpi_enabled: bool
    output_path: str
    average_on_cpu: bool
    performance_disable_training: bool
    performance_disable_communication: bool
    shared_training_loader: Any

    def __init__(self):
        self.max_tick = 0
        self.current_tick = 0
        self.node_container = {}
        self.dataset_label = []
        self.phase = SimulationPhase.INITIALIZING
        self.topology = None

        self.service_container = {}
        self.mpi_enabled = False
        self.output_path = ""

        self.average_on_cpu = True

        self.performance_disable_training = False
        self.performance_disable_communication = False
        self.shared_training_loader = None
