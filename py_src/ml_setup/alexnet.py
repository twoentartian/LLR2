from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from typing import Optional

from py_src.ml_setup_dataset import DatasetSetup, dataset_imagenet1k
from .shared_setup_util import make_setup
from .imagenet_preset import preset_version, imagenet_criterion

import torchvision.models as models

# ---------------------------------------------------------------------------
# AlexNet
# ---------------------------------------------------------------------------

def alexnet_imagenet1k(preset: int = 1, override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    pv: int = preset_version(preset)
    ds = override_dataset if override_dataset is not None else dataset_imagenet1k(preset_version=pv)
    model = models.alexnet(progress=False, weights=None, num_classes=1000)
    return make_setup(model, ModelType.alexnet, ds, 256)
