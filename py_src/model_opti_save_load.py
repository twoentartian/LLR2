from typing import Any, Optional

import torch

stat_dict_key = "state_dict"
model_name_key = "model_name"
dataset_name_key = "dataset_name"


def save_model_state(path: str, state_dict: dict, model_type_name: str, dataset_type_name: str):
    torch.save({stat_dict_key: state_dict, model_name_key: model_type_name, dataset_name_key: dataset_type_name}, path)


def save_optimizer_state(path: str, state_dict: dict, model_type_name: str, dataset_type_name: str):
    torch.save({stat_dict_key: state_dict, model_name_key: model_type_name, dataset_name_key: dataset_type_name}, path)


def load_model_state_file(path: str) -> tuple[dict[str, Any], Optional[str], Optional[str]]:
    data = torch.load(path, weights_only=True)
    return data[stat_dict_key], data.get(model_name_key), data.get(dataset_name_key)

def load_optimizer_state_file(path: str) -> tuple[dict[str, Any], Optional[str], Optional[str]]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data[stat_dict_key], data.get(model_name_key), data.get(dataset_name_key)
