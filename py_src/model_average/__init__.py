"""Public model-averaging interface.

Implementations live in focused modules while this package preserves the
historic ``py_src.model_average`` import path.
"""

from .core import (
    ConservativeModelAverager,
    ModelAverager,
    StandardModelAverager,
    move_model_state_toward,
    move_tensor_toward,
)
from .dfedavgm import DFedAvgMAverager
from .decentralized_fedprox import DecentralizedFedProxAverager
from .dsgt import DSGTModelAverager
from .variance import VarianceCorrectionType, VarianceCorrector


# Keep class identities at the stable public module path for introspection and
# serialization compatibility with existing LLR2 callers.
for _public_class in (
    ModelAverager,
    StandardModelAverager,
    ConservativeModelAverager,
    DFedAvgMAverager,
    DecentralizedFedProxAverager,
    DSGTModelAverager,
    VarianceCorrectionType,
    VarianceCorrector,
):
    _public_class.__module__ = __name__


__all__ = [
    "ConservativeModelAverager",
    "DFedAvgMAverager",
    "DSGTModelAverager",
    "DecentralizedFedProxAverager",
    "ModelAverager",
    "StandardModelAverager",
    "VarianceCorrectionType",
    "VarianceCorrector",
    "move_model_state_toward",
    "move_tensor_toward",
]
