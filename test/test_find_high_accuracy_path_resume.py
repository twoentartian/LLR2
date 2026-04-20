from __future__ import annotations

import copy
import os
import tempfile
import textwrap
import unittest

import torch

from py_src.ml_setup import lenet5_mnist
from py_src.model_opti_save_load import save_model_state
from py_src.util import geodesic_distance
from test.util import make_dummy_mnist
from tool.find_high_accuracy_path import FindHighAccuracyPathRunner, RuntimeParameters, WorkMode


def _clone_state_dict(state_dict):
    output = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            output[key] = value.detach().clone()
        else:
            output[key] = copy.deepcopy(value)
    return output


def _state_dict_value_equal(value_a, value_b) -> bool:
    if torch.is_tensor(value_a) and torch.is_tensor(value_b):
        return torch.equal(value_a, value_b)
    return value_a == value_b


class TestFindHighAccuracyPathResume(unittest.TestCase):
    def test_resume_restores_phase_start_and_tick(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            tick_log_path = os.path.join(temp_dir, "tick_log.txt")
            os.environ["FIND_PATH_TEST_TICK_LOG"] = tick_log_path

            config_path = os.path.join(temp_dir, "resume_config.py")
            with open(config_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(
                    textwrap.dedent(
                        """
                        import os
                        import torch

                        from tool.find_high_accuracy_path.find_parameters import (
                            ParameterGeneral,
                            ParameterMove,
                            ParameterRebuildNorm,
                            ParameterTrain,
                        )


                        def _log_tick(name, tick):
                            with open(os.environ["FIND_PATH_TEST_TICK_LOG"], "a", encoding="utf-8") as fh:
                                fh.write(f"{name}:{tick}\\n")


                        def get_parameter_general(runtime_parameter, ml_setup):
                            _log_tick("general", runtime_parameter.current_tick)
                            output = ParameterGeneral()
                            output.max_tick = 20
                            output.dataloader_worker = 0
                            output.test_dataset_use_whole = False
                            return output


                        def get_parameter_move(runtime_parameter, ml_setup):
                            _log_tick("move", runtime_parameter.current_tick)
                            if runtime_parameter.current_tick != 0:
                                return None
                            output = ParameterMove()
                            output.step_size = 0.0
                            output.adoptive_step_size = 0.0
                            output.ratio_step_size = 0.5
                            output.layer_skip_move = []
                            output.layer_skip_move_keyword = []
                            output.merge_bias_with_weights = False
                            return output


                        def get_parameter_train(runtime_parameter, ml_setup):
                            _log_tick("train", runtime_parameter.current_tick)
                            if runtime_parameter.current_tick != 0:
                                return None
                            output = ParameterTrain()
                            output.train_for_max_rounds = 1
                            output.train_for_min_rounds = 1
                            output.train_until_loss = None
                            output.pretrain_optimizer = False
                            output.pretrain_model_weights = False
                            output.load_existing_optimizer = False
                            return output


                        def get_optimizer_train(runtime_parameter, ml_setup, model_parameter):
                            _log_tick("optimizer", runtime_parameter.current_tick)
                            if runtime_parameter.current_tick != 0:
                                return None
                            return torch.optim.SGD(model_parameter, lr=0.01)


                        def get_parameter_rebuild_norm(runtime_parameter, ml_setup):
                            _log_tick("rebuild", runtime_parameter.current_tick)
                            if runtime_parameter.current_tick != 0:
                                return None
                            output = ParameterRebuildNorm()
                            output.rebuild_norm_for_max_rounds = 0
                            output.rebuild_norm_for_min_rounds = 0
                            output.rebuild_norm_until_loss = 0.0
                            output.rebuild_norm_layer = []
                            output.rebuild_norm_layer_keyword = []
                            return output


                        def get_optimizer_rebuild_norm(runtime_parameter, ml_setup, model_parameter):
                            return None
                        """
                    ).strip()
                    + "\n"
                )

            ml_setup = lenet5_mnist(override_dataset=make_dummy_mnist(num_samples=8, return_pil=False))
            start_state = _clone_state_dict(ml_setup.model.state_dict())
            probe_key = next(
                key
                for key, value in start_state.items()
                if torch.is_tensor(value) and value.dtype.is_floating_point
            )
            end_state = _clone_state_dict(start_state)
            end_state[probe_key] = end_state[probe_key] + 0.2

            start_model_path = os.path.join(temp_dir, "start", "0.model.pt")
            end_model_path = os.path.join(temp_dir, "end", "0.model.pt")
            os.makedirs(os.path.dirname(start_model_path), exist_ok=True)
            os.makedirs(os.path.dirname(end_model_path), exist_ok=True)
            save_model_state(
                start_model_path,
                start_state,
                ml_setup.model_type.name,
                ml_setup.dataset_type.name,
            )
            save_model_state(
                end_model_path,
                end_state,
                ml_setup.model_type.name,
                ml_setup.dataset_type.name,
            )

            runtime_parameter = RuntimeParameters()
            runtime_parameter.start_and_end_point_for_paths = [(start_model_path, end_model_path)]
            runtime_parameter.use_cpu = True
            runtime_parameter.use_amp = False
            runtime_parameter.work_mode = WorkMode.to_certain_model
            runtime_parameter.output_folder_path = output_dir
            runtime_parameter.total_cpu_count = 1
            runtime_parameter.worker_count = 1
            runtime_parameter.save_ticks = None
            runtime_parameter.save_interval = 1
            runtime_parameter.save_format = "none"
            runtime_parameter.config_file_path = config_path
            runtime_parameter.dataset_type = None
            runtime_parameter.model_type = None
            runtime_parameter.pytorch_preset_version = 1
            runtime_parameter.store_top_accuracy_model_count = 1
            runtime_parameter.checkpoint_interval = 1
            runtime_parameter.task_name = None # type: ignore
            runtime_parameter.silence_mode = True
            runtime_parameter.across_vs_lr_policy = "var"
            runtime_parameter.linear_interpolation_points_size = 0
            runtime_parameter.linear_interpolation_dataset_size = 0
            runtime_parameter.variance_sphere_file_path = None # type: ignore
            runtime_parameter.variance_sphere_model = None
            runtime_parameter.stop_when_training_loss_exceeds = None # type: ignore
            runtime_parameter.debug_check_config_mode = False
            runtime_parameter.test_dataset_use_whole = False
            runtime_parameter.verbose = False
            runtime_parameter.service_test_accuracy_loss_disable = True
            runtime_parameter.service_test_accuracy_loss_interval = 1
            runtime_parameter.service_test_accuracy_loss_batch_size = 2
            runtime_parameter.service_cosine_similarity_disable = True
            runtime_parameter.service_cosine_similarity_ref_model = None # type: ignore
            runtime_parameter.enable_profiler = False

            runner = FindHighAccuracyPathRunner()
            runner.setup(0, runtime_parameter)
            assert runner.model is not None
            assert runner.current_phase_start_model_stat is not None
            assert runner.runtime_parameter is not None

            current_state = _clone_state_dict(runner.model.state_dict())
            current_state[probe_key] = current_state[probe_key] + 0.05
            runner.model.load_state_dict(current_state)
            runner.runtime_parameter.current_tick = 5
            runner._maybe_save_checkpoint()
            assert runner.latest_checkpoint_path is not None

            with open(tick_log_path, "w", encoding="utf-8") as file_handle:
                file_handle.write("")

            resumed_runtime_parameter = RuntimeParameters()
            resumed_runtime_parameter.start_and_end_point_for_paths = [(start_model_path, end_model_path)]
            resumed_runtime_parameter.use_cpu = True
            resumed_runtime_parameter.use_amp = False
            resumed_runtime_parameter.work_mode = WorkMode.to_certain_model
            resumed_runtime_parameter.output_folder_path = output_dir
            resumed_runtime_parameter.total_cpu_count = 1
            resumed_runtime_parameter.worker_count = 1
            resumed_runtime_parameter.save_ticks = None
            resumed_runtime_parameter.save_interval = 1
            resumed_runtime_parameter.save_format = "none"
            resumed_runtime_parameter.config_file_path = config_path
            resumed_runtime_parameter.dataset_type = None
            resumed_runtime_parameter.model_type = None
            resumed_runtime_parameter.pytorch_preset_version = 1
            resumed_runtime_parameter.store_top_accuracy_model_count = 1
            resumed_runtime_parameter.checkpoint_interval = 1
            resumed_runtime_parameter.task_name = None # type: ignore
            resumed_runtime_parameter.silence_mode = True
            resumed_runtime_parameter.across_vs_lr_policy = "var"
            resumed_runtime_parameter.linear_interpolation_points_size = 0
            resumed_runtime_parameter.linear_interpolation_dataset_size = 0
            resumed_runtime_parameter.variance_sphere_file_path = None # type: ignore
            resumed_runtime_parameter.variance_sphere_model = None
            resumed_runtime_parameter.stop_when_training_loss_exceeds = None # type: ignore
            resumed_runtime_parameter.debug_check_config_mode = False
            resumed_runtime_parameter.test_dataset_use_whole = False
            resumed_runtime_parameter.verbose = False
            resumed_runtime_parameter.service_test_accuracy_loss_disable = True
            resumed_runtime_parameter.service_test_accuracy_loss_interval = 1
            resumed_runtime_parameter.service_test_accuracy_loss_batch_size = 2
            resumed_runtime_parameter.service_cosine_similarity_disable = True
            resumed_runtime_parameter.service_cosine_similarity_ref_model = None # type: ignore
            resumed_runtime_parameter.enable_profiler = False

            resumed_runner = FindHighAccuracyPathRunner()
            resumed_runner.setup(0, resumed_runtime_parameter, runner.latest_checkpoint_path)

            assert resumed_runner.runtime_parameter is not None
            assert resumed_runner.parameter_move is not None
            assert resumed_runner.current_phase_start_model_stat is not None
            assert resumed_runner.ratio_step_size is not None
            assert resumed_runner.end_model_stat_dict is not None
            assert resumed_runner.model is not None

            self.assertEqual(resumed_runner.runtime_parameter.current_tick, 5)
            self.assertEqual(resumed_runner.parameter_move.ratio_step_size, 0.5)
            self.assertTrue(
                _state_dict_value_equal(
                    resumed_runner.current_phase_start_model_stat[probe_key],
                    start_state[probe_key],
                )
            )

            expected_distance = geodesic_distance(
                resumed_runner.current_phase_start_model_stat[probe_key],
                resumed_runner.end_model_stat_dict[probe_key],
            )
            current_distance = geodesic_distance(
                resumed_runner.model.state_dict()[probe_key],
                resumed_runner.end_model_stat_dict[probe_key],
            )
            assert expected_distance is not None
            assert current_distance is not None
            expected_ratio_step = expected_distance.item() * 0.5
            current_ratio_step = current_distance.item() * 0.5

            self.assertAlmostEqual(
                resumed_runner.ratio_step_size[probe_key],
                expected_ratio_step,
                places=6,
            )
            self.assertNotAlmostEqual(current_ratio_step, expected_ratio_step, places=6)

            with open(tick_log_path, "r", encoding="utf-8") as file_handle:
                tick_log_entries = {line.strip() for line in file_handle if line.strip()}

            self.assertIn("general:5", tick_log_entries)
            self.assertIn("move:5", tick_log_entries)
            self.assertIn("train:5", tick_log_entries)
            self.assertIn("rebuild:5", tick_log_entries)
            self.assertIn("optimizer:5", tick_log_entries)
            self.assertIn("optimizer:0", tick_log_entries)


if __name__ == "__main__":
    unittest.main()
