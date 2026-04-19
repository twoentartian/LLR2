"""Minimal SimulationPhase enum and RuntimeParameters stub.

These are kept for compatibility with the service interfaces originally
designed for DFL_torch's simulator. The services use
``initialize_without_runtime_parameters`` / ``trigger_without_runtime_parameters``
in the stand-alone tool; the ``initialize`` / ``trigger`` variants remain for
future simulator integration.
"""

from enum import Enum


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
    def __init__(self):
        self.max_tick:int = 0
        self.current_tick:int = 0
        self.node_container = None
        self.dataset_label = None
        self.phase = SimulationPhase.INITIALIZING
        self.topology = None

        self.service_container = {}
        self.mpi_enabled = None
        self.output_path = None
