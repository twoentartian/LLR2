import argparse
import torch
import os
import sys
from torch.utils.data import DataLoader
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, util


def testing_model(model, current_ml_setup, test_training, batch_size):
    testing_dataset = current_ml_setup.testing_data
    training_dataset = current_ml_setup.training_data
    criterion = current_ml_setup.criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if current_ml_setup.override_testing_dataset_loader is not None:
        dataloader_test = current_ml_setup.override_testing_dataset_loader
    else:
        dataloader_test = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    if test_training:
        if current_ml_setup.override_training_dataset_loader is not None:
            dataloader_train = current_ml_setup.override_training_dataset_loader
        else:
            dataloader_train = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    else:
        dataloader_train = None

    model.eval()
    model.to(device)
    test_loss = 0
    train_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        if current_ml_setup.override_evaluation_step_function is not None:
            val_loss, val_correct, val_count = 0.0, 0.0, 0
            for batch_idx, batch in enumerate(dataloader_test):
                output = current_ml_setup.override_evaluation_step_function(batch_idx, batch, model, None, None, current_ml_setup)
                val_loss += output.loss_value * output.sample_count
                val_correct += output.correct_count
                val_count += output.sample_count
            test_loss = val_loss / val_count
            test_accuracy = val_correct / val_count
        else:
            for batch_idx, (data, label) in enumerate(dataloader_test):
                print(f"test batch_idx: {batch_idx}")
                data, label = data.to(device), label.to(device)
                outputs = model(data)
                loss = criterion(outputs, label)
                test_loss += loss.item() * data.size(0)
                _, predicted = outputs.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()
            test_loss = test_loss / total
            test_accuracy = correct / total

        if dataloader_train is not None:
            if current_ml_setup.override_evaluation_step_function is not None:
                val_loss, val_correct, val_count = 0.0, 0.0, 0
                for batch_idx, batch in enumerate(dataloader_train):
                    output = current_ml_setup.override_evaluation_step_function(batch_idx, batch, model, None, None, current_ml_setup)
                    val_loss += output.loss_value * output.sample_count
                    val_correct += output.correct_count
                    val_count += output.sample_count
                train_loss = val_loss / val_count
                train_accuracy = val_correct / val_count
            else:
                for batch_idx, (data, label) in enumerate(dataloader_train):
                    print(f"train batch_idx: {batch_idx}")
                    data, label = data.to(device), label.to(device)
                    if current_ml_setup.mixup_fn is not None:
                        data, label = current_ml_setup.mixup_fn(data, label)
                    outputs = model(data)
                    loss = criterion(outputs, label)
                    train_loss += loss.item() * data.size(0)
                    _, predicted = outputs.max(1)
                    total += label.size(0)
                    correct += predicted.eq(label).sum().item()
                train_loss = train_loss / total
                train_accuracy = correct / total
        else:
            train_loss = 0
            train_accuracy = 0
    return test_loss, test_accuracy, train_loss, train_accuracy

def get_layer_variances(state_dict):
    output = {}
    for name, param in state_dict.items():
        if torch.is_tensor(param) and param.dtype.is_floating_point:
            var = param.var().item()
            output[name] = var
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure model test accuracy and loss.')
    parser.add_argument("model_file", type=str)
    parser.add_argument("-m", "--model_type", type=str, default="auto")
    parser.add_argument("-d", "--dataset_type", type=str, default="auto")
    parser.add_argument("-t", "--training", action="store_true")
    parser.add_argument("-P", "--torch_preset_version", type=int, default=None, help='specify the pytorch data training preset version')
    parser.add_argument("-b", "--batch_size", type=int, default=100, help='batch size')

    args = parser.parse_args()

    model_file_path = args.model_file
    model_type_from_cli = args.model_type

    model_stat, model_name, dataset_name = util.load_model_state_file(model_file_path)
    if model_type_from_cli == "auto":
        model_type = model_name
    else:
        model_type = model_type_from_cli
    assert model_type is not None, "model_type is None"

    if dataset_name is None:
        if args.dataset_type == "auto":
            current_ml_setup = ml_setup.get_ml_setup_from_config(model_type, dataset_type="default", pytorch_preset_version=args.torch_preset_version)
        else:
            current_ml_setup = ml_setup.get_ml_setup_from_config(model_type, dataset_type=args.dataset_type, pytorch_preset_version=args.torch_preset_version)
    else:
        if args.dataset_type == "auto":
            current_ml_setup = ml_setup.get_ml_setup_from_config(model_type, dataset_type=dataset_name, pytorch_preset_version=args.torch_preset_version)
        else:
            if dataset_name != args.dataset_type:
                print(f"WARNING: dataset_name in CLI({args.dataset_type}) and in model state file ({dataset_name}) mismatch.")
                print(f"dataset type override to {args.dataset_type}")
            current_ml_setup = ml_setup.get_ml_setup_from_config(model_type, dataset_type=args.dataset_type, pytorch_preset_version=args.torch_preset_version)

    if not os.path.exists(model_file_path):
        print(f"file not found. {model_file_path}")
    model = current_ml_setup.model
    model.load_state_dict(model_stat)

    test_loss, test_accuracy, train_loss, train_accuracy = testing_model(model, current_ml_setup, args.training, args.batch_size)
    layer_variances = get_layer_variances(model_stat)
    print(f"test loss={test_loss}, test acc={test_accuracy}")
    print(f"train loss={train_loss}, train acc={train_accuracy}")
    if test_accuracy * train_accuracy >0.001:
        with open(f"{model_file_path}.txt", "w") as f:
            f.write(f"test loss={test_loss}, test acc={test_accuracy}\n")
            f.write(f"train loss={train_loss}, train acc={train_accuracy}\n")
            f.write("\nLayer Variance List:\n")
            for layer_name, layer_var in layer_variances.items():
                f.write(f"layer {layer_name}: {layer_var}\n")
