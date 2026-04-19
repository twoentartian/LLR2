"""Service base class — kept for simulator interface compatibility."""
from abc import ABC, abstractmethod

from py_src.simulation_runtime_parameters import RuntimeParameters


class Service(ABC):
    @staticmethod
    @abstractmethod
    def get_service_name() -> str:
        pass

    @abstractmethod
    def initialize(self, parameters: RuntimeParameters, output_path, *args, **kwargs):
        pass

    @abstractmethod
    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        pass

    @abstractmethod
    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, *args, **kwargs):
        pass
