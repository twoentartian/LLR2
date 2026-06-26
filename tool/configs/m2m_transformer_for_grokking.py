import os
import sys

import torch

from find_high_accuracy_path_v2.find_parameters import (
    ParameterGeneral,
    ParameterMove,
    ParameterRebuildNorm,
    ParameterTrain,
)
from find_high_accuracy_path_v2.runtime_parameters import RuntimeParameters

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src.ml_setup import MLSetup, ModelType


supported_model_type_name = ModelType.transformer_for_grokking

FDF_PHASE_TIME = 400

STEP_SIZE = 0.0
ADOPTIVE_STEP_SIZE = 0.002
RATIO_STEP_SIZE = 0.002
TRAIN_FOR_MAX_ROUNDS = 100
TRAIN_FOR_MIN_ROUNDS = 10
TRAIN_UNTIL_LOSS = 0.001
OPTIMIZER_LR = 0.0001

FDF_PHASE_PREFIXES = ['embedding',
 'decoder.blocks.0.self_attn.attn_heads.0.Wq',
 'decoder.blocks.0.self_attn.attn_heads.0.Wk',
 'decoder.blocks.0.self_attn.attn_heads.0.Wv',
 'decoder.blocks.0.self_attn.attn_heads.1.Wq',
 'decoder.blocks.0.self_attn.attn_heads.1.Wk',
 'decoder.blocks.0.self_attn.attn_heads.1.Wv',
 'decoder.blocks.0.self_attn.attn_heads.2.Wq',
 'decoder.blocks.0.self_attn.attn_heads.2.Wk',
 'decoder.blocks.0.self_attn.attn_heads.2.Wv',
 'decoder.blocks.0.self_attn.attn_heads.3.Wq',
 'decoder.blocks.0.self_attn.attn_heads.3.Wk',
 'decoder.blocks.0.self_attn.attn_heads.3.Wv',
 'decoder.blocks.0.self_attn.Wo',
 'decoder.blocks.0.self_attn_norm',
 'decoder.blocks.0.ffn.ffn',
 'decoder.blocks.0.ffn_norm',
 'decoder.blocks.1.self_attn.attn_heads.0.Wq',
 'decoder.blocks.1.self_attn.attn_heads.0.Wk',
 'decoder.blocks.1.self_attn.attn_heads.0.Wv',
 'decoder.blocks.1.self_attn.attn_heads.1.Wq',
 'decoder.blocks.1.self_attn.attn_heads.1.Wk',
 'decoder.blocks.1.self_attn.attn_heads.1.Wv',
 'decoder.blocks.1.self_attn.attn_heads.2.Wq',
 'decoder.blocks.1.self_attn.attn_heads.2.Wk',
 'decoder.blocks.1.self_attn.attn_heads.2.Wv',
 'decoder.blocks.1.self_attn.attn_heads.3.Wq',
 'decoder.blocks.1.self_attn.attn_heads.3.Wk',
 'decoder.blocks.1.self_attn.attn_heads.3.Wv',
 'decoder.blocks.1.self_attn.Wo',
 'decoder.blocks.1.self_attn_norm',
 'decoder.blocks.1.ffn.ffn',
 'decoder.blocks.1.ffn_norm',
 'linear']

FDF_ALWAYS_SKIP_KEYWORDS = []

FDF_ALWAYS_SKIP_LAYERS = ['position_encoding', 'self_attn_mask']


def _build_fdf_move_parameter(phase_index: int) -> ParameterMove | None:
    if phase_index < 0 or phase_index >= len(FDF_PHASE_PREFIXES):
        return None

    output = ParameterMove()
    test_weights_keyword = ['running_mean', 'running_var', 'num_batches_tracked', 'ema']

    output.step_size = 0
    output.adoptive_step_size = 0.001
    output.ratio_step_size = 0.002
    output.layer_skip_move = FDF_ALWAYS_SKIP_LAYERS
    output.layer_skip_move_keyword = (
        FDF_PHASE_PREFIXES[phase_index + 1:] + test_weights_keyword
    )
    output.merge_bias_with_weights = False
    return output


def get_parameter_general(runtime_parameter: RuntimeParameters, ml_setup: MLSetup):
    output = ParameterGeneral()
    if ml_setup.model_type == supported_model_type_name:
        output.max_tick = FDF_PHASE_TIME * (len(FDF_PHASE_PREFIXES)+2)
        output.dataloader_worker = 8
        output.test_dataset_use_whole = True
    else:
        raise NotImplemented
    return output


def get_parameter_move(runtime_parameter: RuntimeParameters, ml_setup: MLSetup):
    if ml_setup.model_type == supported_model_type_name:
        if runtime_parameter.current_tick % FDF_PHASE_TIME != 0:
            return None
        phase_index = runtime_parameter.current_tick // FDF_PHASE_TIME
        output = _build_fdf_move_parameter(phase_index)
        if output is None:
            return None
    else:
        raise NotImplemented
    return output


def get_parameter_train(runtime_parameter: RuntimeParameters, ml_setup: MLSetup):
    output = ParameterTrain()
    if ml_setup.model_type == supported_model_type_name:
        if runtime_parameter.current_tick == 0:
            output.train_for_max_rounds = TRAIN_FOR_MAX_ROUNDS
            output.train_for_min_rounds = TRAIN_FOR_MIN_ROUNDS
            output.train_until_loss = TRAIN_UNTIL_LOSS
            output.pretrain_optimizer = False
            output.load_existing_optimizer = False
        else:
            return None
    else:
        raise NotImplemented
    return output


def get_optimizer_train(runtime_parameter: RuntimeParameters, ml_setup: MLSetup, model_parameter):
    if ml_setup.model_type == supported_model_type_name:
        if runtime_parameter.current_tick == 0:
            optimizer = torch.optim.Adam(model_parameter, lr=OPTIMIZER_LR)
        else:
            return None
    else:
        raise NotImplemented
    return optimizer


def get_parameter_rebuild_norm(runtime_parameter: RuntimeParameters, ml_setup: MLSetup):
    output = ParameterRebuildNorm()
    if ml_setup.model_type == supported_model_type_name:
        if runtime_parameter.current_tick == 0:
            output.rebuild_norm_for_max_rounds = 0
            output.rebuild_norm_for_min_rounds = 0
            output.rebuild_norm_until_loss = 10
            output.rebuild_norm_layer = []
            output.rebuild_norm_layer_keyword = []
        else:
            return None
    else:
        raise NotImplemented
    return output


def get_optimizer_rebuild_norm(runtime_parameter: RuntimeParameters, ml_setup: MLSetup, model_parameter):
    if ml_setup.model_type == supported_model_type_name:
        if runtime_parameter.current_tick == 0:
            optimizer = None
        else:
            return None
    else:
        raise NotImplemented
    return optimizer
