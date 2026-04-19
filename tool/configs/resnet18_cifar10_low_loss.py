import torch
import sys
import os

from find_high_accuracy_path_v2.find_parameters import ParameterMove, ParameterTrain, ParameterRebuildNorm, ParameterGeneral
from find_high_accuracy_path_v2.runtime_parameters import RuntimeParameters

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src.ml_setup import MlSetup

model_name = 'resnet18_bn'

def get_parameter_general(runtime_parameter: RuntimeParameters, ml_setup: MlSetup):
    output = ParameterGeneral()
    if ml_setup.model_name == model_name:
        output.max_tick = 8000
        output.dataloader_worker = 8
        output.test_dataset_use_whole = True
    else:
        raise NotImplemented
    return output

def get_parameter_move(runtime_parameter: RuntimeParameters, ml_setup: MlSetup):
    output = ParameterMove()
    if ml_setup.model_name == model_name:
        if runtime_parameter.current_tick == 0:
            output.step_size = 0
            output.adoptive_step_size = 0.002
            output.layer_skip_move = []
            output.layer_skip_move_keyword = ["layer1", "layer2", "layer3", "layer4", "num_batches_tracked", "running_mean", "running_var", "fc"]
            output.merge_bias_with_weights = False
        elif runtime_parameter.current_tick == 1000:
            output.step_size = 0
            output.adoptive_step_size = 0.002
            output.layer_skip_move = []
            output.layer_skip_move_keyword = ["layer2", "layer3", "layer4", "num_batches_tracked", "running_mean", "running_var", "fc"]
            output.merge_bias_with_weights = False
        elif runtime_parameter.current_tick == 2000:
            output.step_size = 0
            output.adoptive_step_size = 0.002
            output.layer_skip_move = []
            output.layer_skip_move_keyword = ["layer3", "layer4", "num_batches_tracked", "running_mean", "running_var", "fc"]
            output.merge_bias_with_weights = False
        elif runtime_parameter.current_tick == 3000:
            output.step_size = 0
            output.adoptive_step_size = 0.002
            output.layer_skip_move = []
            output.layer_skip_move_keyword = ["layer4", "num_batches_tracked", "running_mean", "running_var", "fc"]
            output.merge_bias_with_weights = False
        elif runtime_parameter.current_tick == 4000:
            output.step_size = 0
            output.adoptive_step_size = 0.002
            output.layer_skip_move = []
            output.layer_skip_move_keyword = ["num_batches_tracked", "running_mean", "running_var", "fc"]
            output.merge_bias_with_weights = False
        elif runtime_parameter.current_tick == 5000:
            output.step_size = 0
            output.adoptive_step_size = 0.002
            output.layer_skip_move = []
            output.layer_skip_move_keyword = ["num_batches_tracked", "running_mean", "running_var"]
            output.merge_bias_with_weights = False
        else:
            return None
    else:
        raise NotImplemented
    return output


def get_parameter_train(runtime_parameter: RuntimeParameters, ml_setup: MlSetup):
    output = ParameterTrain()
    if ml_setup.model_name == model_name:
        if runtime_parameter.current_tick == 0:
            output.train_for_max_rounds = 20000
            output.train_for_min_rounds = 20
            output.train_until_loss = 0.005
            output.pretrain_optimizer = True
            output.load_existing_optimizer = False
        else:
            return None
    else:
        raise NotImplemented
    return output

def get_optimizer_train(runtime_parameter: RuntimeParameters, ml_setup: MlSetup, model_parameter):
    if ml_setup.model_name == model_name:
        if runtime_parameter.current_tick == 0:
            optimizer = torch.optim.SGD(model_parameter, lr=0.001)
        else:
            return None
    else:
        raise NotImplemented
    return optimizer

def get_parameter_rebuild_norm(runtime_parameter: RuntimeParameters, ml_setup: MlSetup):
    output = ParameterRebuildNorm()
    if ml_setup.model_name == model_name:
        if runtime_parameter.current_tick == 0:
            output.rebuild_norm_for_max_rounds = 0
            output.rebuild_norm_for_min_rounds = 0
            output.rebuild_norm_until_loss = 0
            output.rebuild_norm_layer = []
            output.rebuild_norm_layer_keyword = ['bn']
        else:
            return None
    else:
        raise NotImplemented
    return output

def get_optimizer_rebuild_norm(runtime_parameter: RuntimeParameters, ml_setup: MlSetup, model_parameter):
    if ml_setup.model_name == model_name:
        if runtime_parameter.current_tick == 0:
            optimizer = torch.optim.SGD(model_parameter, lr=0.001, momentum=0.9, weight_decay=5e-4)
        else:
            return None
    else:
        raise NotImplemented
    return optimizer