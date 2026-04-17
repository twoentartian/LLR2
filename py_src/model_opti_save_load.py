import torch


def save_model_state(path: str, state_dict: dict, model_type_name: str, dataset_type_name: str):
    torch.save({"model_state": state_dict, "model_type": model_type_name, "dataset_type": dataset_type_name}, path)


def save_optimizer_state(path: str, state_dict: dict, model_type_name: str, dataset_type_name: str):
    torch.save({"optimizer_state": state_dict, "model_type": model_type_name, "dataset_type": dataset_type_name}, path)


def load_model_state_file(path: str):
    data = torch.load(path, weights_only=True)
    return data["model_state"], data.get("model_type"), data.get("dataset_type")

def load_optimizer_state_file(path: str):
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["optimizer_state"], data.get("model_type"), data.get("dataset_type")