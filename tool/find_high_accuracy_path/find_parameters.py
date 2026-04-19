"""Parameter dataclasses for the path-finding algorithm."""

from typing import List, Optional


class Parameter:
    def validate(self):
        pass  # subclasses may override


class ParameterGeneral(Parameter):
    max_tick: Optional[int] = None
    dataloader_worker: Optional[int] = None
    dataloader_prefetch_factor: Optional[int] = None
    test_dataset_use_whole: Optional[bool] = None   # False by default
    split_test_val: Optional[float] = None


class ParameterMove(Parameter):
    step_size: Optional[float] = None
    adoptive_step_size: Optional[float] = None
    ratio_step_size: Optional[float] = None
    layer_skip_move: Optional[List[str]] = None
    layer_skip_move_keyword: Optional[List[str]] = None
    merge_bias_with_weights: Optional[bool] = None

    # Layer-norm compensate
    layer_compensate_x2: Optional[List[str]] = None
    layer_compensate_x2_keyword: Optional[List[str]] = None

    # Attention layers
    layer_attention: Optional[List[str]] = None
    layer_attention_keyword: Optional[List[str]] = None
    layer_attention_policy: Optional[str] = None  # 'none', 'ignore_kv'

    def fill_default(self):
        if self.layer_compensate_x2 is None:
            self.layer_compensate_x2 = []
        if self.layer_compensate_x2_keyword is None:
            self.layer_compensate_x2_keyword = []
        if self.layer_attention is None:
            self.layer_attention = []
        if self.layer_attention_keyword is None:
            self.layer_attention_keyword = []
        if self.layer_attention_policy is None:
            self.layer_attention_policy = 'none'


class ParameterTrain(Parameter):
    train_for_max_rounds: Optional[int] = None
    train_until_loss: Optional[float] = None
    train_for_min_rounds: Optional[int] = None
    pretrain_optimizer: Optional[bool] = None
    pretrain_model_weights: Optional[bool] = None
    pretrain_iterations: Optional[int] = None
    load_existing_optimizer: Optional[bool] = None

    def fill_default(self):
        if self.pretrain_iterations is None:
            self.pretrain_iterations = 100


class ParameterRebuildNorm(Parameter):
    rebuild_norm_for_max_rounds: Optional[int] = None
    rebuild_norm_for_min_rounds: Optional[int] = None
    rebuild_norm_until_loss: Optional[float] = None
    rebuild_norm_layer: Optional[List[str]] = None
    rebuild_norm_layer_keyword: Optional[List[str]] = None
    rebuild_norm_use_initial_norm_weights: bool = False
    rebuild_norm_use_start_model_norm_weights: bool = False
