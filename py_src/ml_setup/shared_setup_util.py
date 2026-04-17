import torch.nn as nn

from py_src.adapters import StandardAdapter
from py_src.ml_setup.ml_setup import MLSetup

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_setup(model, model_type, dataset_setup, batch_size, has_normalization=True,
                criterion=None, clip_grad_norm=None) -> MLSetup:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    adapter = StandardAdapter(model, criterion, clip_grad_norm=clip_grad_norm)
    return MLSetup(
        model=model,
        adapter=adapter,
        model_type=model_type,
        training_data=dataset_setup.train_data,
        testing_data=dataset_setup.valdation_data,
        dataset_type=dataset_setup.dataset_type,
        default_batch_size=batch_size,
        has_normalization_layer=has_normalization,
    )