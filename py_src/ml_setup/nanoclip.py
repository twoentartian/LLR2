from __future__ import annotations

from typing import Optional

from transformers import AutoTokenizer

from py_src.adapters import LightningAdapter
from py_src.ml_setup.ml_setup import ApplicationType, MLSetup
from py_src.ml_setup_dataset import CollateFlickr, DatasetSetup, dataset_flickr30k
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_model.nanoclip import ContrastiveLoss, NanoCLIP


def nanoclip_flickr30k_default(override_dataset: Optional[DatasetSetup] = None) -> MLSetup:
    txt_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dataset = override_dataset if override_dataset is not None else dataset_flickr30k()

    model = NanoCLIP(
        txt_model=txt_model_name,
        img_model="dinov2_vits14",
        unfreeze_n_blocks=4,
        embed_size=64,
        lr=0.001,
        freeze_encoder=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(txt_model_name)
    criterion = ContrastiveLoss(temperature=0.05)

    output_ml_setup = MLSetup(
        model=model,
        adapter=LightningAdapter(model),
        model_type=ModelType.nanoclip_default,
        training_data=dataset.train_data,
        testing_data=dataset.valdation_data,
        dataset_type=dataset.dataset_type,
        default_batch_size=128,
        default_collate_fn=CollateFlickr(tokenizer, max_length=80, captions_to_use="all"),
        default_collate_fn_val=CollateFlickr(tokenizer, max_length=80, captions_to_use="first"),
        has_normalization_layer=True,
        application_type=ApplicationType.clip,
    )
    # Keep a DFL_torch-style attribute around for any callers that still look for it.
    setattr(output_ml_setup, "criterion", criterion)
    return output_ml_setup
