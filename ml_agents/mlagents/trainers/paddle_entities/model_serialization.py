from typing import Tuple
import threading
import os
import paddle

from mlagents.trainers.settings import SerializationSettings
from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)


class exporting_to_onnx:
    """
    Set this context by calling
    ```
    with exporting_to_onnx():
    ```
    Within this context, the variable exporting_to_onnx.is_exporting() will be true.
    This implementation is thread safe.
    """

    _local_data = threading.local()
    _local_data._is_exporting = False
    _lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        self._local_data._is_exporting = True

    def __exit__(self, *args):
        self._local_data._is_exporting = False
        self._lock.release()

    @staticmethod
    def is_exporting():
        if not hasattr(exporting_to_onnx._local_data, "_is_exporting"):
            return False
        return exporting_to_onnx._local_data._is_exporting


class TensorNames:
    batch_size_placeholder = "batch_size"
    sequence_length_placeholder = "sequence_length"
    vector_observation_placeholder = "vector_observation"
    recurrent_in_placeholder = "recurrent_in"
    visual_observation_placeholder_prefix = "visual_observation_"
    observation_placeholder_prefix = "obs_"
    previous_action_placeholder = "prev_action"
    action_mask_placeholder = "action_masks"
    random_normal_epsilon_placeholder = "epsilon"

    value_estimate_output = "value_estimate"
    recurrent_output = "recurrent_out"
    memory_size = "memory_size"
    version_number = "version_number"

    continuous_action_output_shape = "continuous_action_output_shape"
    discrete_action_output_shape = "discrete_action_output_shape"
    continuous_action_output = "continuous_actions"
    discrete_action_output = "discrete_actions"
    deterministic_continuous_action_output = "deterministic_continuous_actions"
    deterministic_discrete_action_output = "deterministic_discrete_actions"

    is_continuous_control_deprecated = "is_continuous_control"
    action_output_deprecated = "action"
    action_output_shape_deprecated = "action_output_shape"

    @staticmethod
    def get_visual_observation_name(index: int) -> str:
        """
        Returns the name of the visual observation with a given index
        """
        return TensorNames.visual_observation_placeholder_prefix + str(index)

    @staticmethod
    def get_observation_name(index: int) -> str:
        """
        Returns the name of the observation with a given index
        """
        return TensorNames.observation_placeholder_prefix + str(index)


class ModelSerializer:
    def __init__(self, policy):
        self.policy = policy
        observation_specs = self.policy.behavior_spec.observation_specs
        batch_dim = [1]
        seq_len_dim = [1]
        num_obs = len(observation_specs)

        dummy_obs = [
            paddle.zeros(
                batch_dim + list(ModelSerializer._get_onnx_shape(obs_spec.shape)),
                dtype='float32'
            )
            for obs_spec in observation_specs
        ]

        dummy_masks = paddle.ones(
            batch_dim + [sum(self.policy.behavior_spec.action_spec.discrete_branches)],
            dtype='float32'
        )
        dummy_memories = paddle.zeros(
            batch_dim + seq_len_dim + [self.policy.export_memory_size],
            dtype='float32'
        )

        self.dummy_input = (dummy_obs, dummy_masks, dummy_memories)

        self.input_names = [TensorNames.get_observation_name(i) for i in range(num_obs)]
        self.input_names += [TensorNames.action_mask_placeholder]
        if self.policy.export_memory_size > 0:
            self.input_names += [TensorNames.recurrent_in_placeholder]

        self.output_names = [TensorNames.version_number, TensorNames.memory_size]
        if self.policy.behavior_spec.action_spec.continuous_size > 0:
            self.output_names += [
                TensorNames.continuous_action_output,
                TensorNames.continuous_action_output_shape,
                TensorNames.deterministic_continuous_action_output,
            ]
        if self.policy.behavior_spec.action_spec.discrete_size > 0:
            self.output_names += [
                TensorNames.discrete_action_output,
                TensorNames.discrete_action_output_shape,
                TensorNames.deterministic_discrete_action_output,
            ]

        if self.policy.export_memory_size > 0:
            self.output_names += [TensorNames.recurrent_output]

        self.input_specs = []
        pre_obs=[]
        for i, obs_tensor in enumerate(dummy_obs):
            shape = list(obs_tensor.shape)
            shape[0] = None
            pre_obs.append(
                paddle.static.InputSpec(shape=shape, dtype='float32', name=self.input_names[i])
            )
        self.input_specs.append(pre_obs)

        mask_shape = list(dummy_masks.shape)
        mask_shape[0] = None
        self.input_specs.append(
            paddle.static.InputSpec(shape=mask_shape, dtype='float32', name=TensorNames.action_mask_placeholder)
        )

        if self.policy.export_memory_size > 0:
            mem_shape = list(dummy_memories.shape)
            mem_shape[0] = None
            self.input_specs.append(
                paddle.static.InputSpec(shape=mem_shape, dtype='float32', name=TensorNames.recurrent_in_placeholder)
            )

    @staticmethod
    def _get_onnx_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Converts the shape of an observation to be compatible with the NCHW format
        of ONNX
        """
        if len(shape) == 3:
            return shape[0], shape[1], shape[2]
        return shape

    def _rename_unity_outputs(self, onnx_path: str) -> None:
        """Rename ONNX graph outputs to names Unity expects."""
        try:
            import onnx
        except ImportError:
            logger.warning(
                "onnx 未安装，无法对导出的 ONNX 输出重命名。运行 pip install onnx 可开启此功能。"
            )
            return
        if not os.path.isfile(onnx_path):
            return

        model = onnx.load(onnx_path)
        g = model.graph

        current_outs = list(g.output)
        n = min(len(current_outs), len(self.output_names))
        if n == 0:
            return

        mapping = {
            current_outs[i].name: self.output_names[i]
            for i in range(n)
            if current_outs[i].name != self.output_names[i]
        }
        if not mapping:
            onnx.save(model, onnx_path)
            return

        def replace_all(old_name: str, new_name: str) -> None:
            for out in g.output:
                if out.name == old_name:
                    out.name = new_name
            for vi in g.value_info:
                if vi.name == old_name:
                    vi.name = new_name
            for node in g.node:
                for i, s in enumerate(node.input):
                    if s == old_name:
                        node.input[i] = new_name
                for i, s in enumerate(node.output):
                    if s == old_name:
                        node.output[i] = new_name

        tmp_prefix = "___unity_tmp_"
        for old in mapping:
            safe = old.replace("/", "_").replace(":", "_")[:50]
            tmp = tmp_prefix + safe
            replace_all(old, tmp)
        for old, new in mapping.items():
            safe = old.replace("/", "_").replace(":", "_")[:50]
            tmp = tmp_prefix + safe
            replace_all(tmp, new)

        onnx.save(model, onnx_path)

    def export_policy_model(self, output_filepath: str) -> None:
        """
        Exports a Paddle model for a Policy to .onnx format for Unity embedding.

        :param output_filepath: file path to output the model (without file suffix)
        """
        model_output_path = f"{output_filepath}"
        logger.debug(f"Converting to {model_output_path}")
        self.policy.actor.eval()
        paddle.save(self.policy.actor.state_dict(), model_output_path)

        with exporting_to_onnx():
            paddle.onnx.export(
                self.policy.actor,
                output_filepath,
                input_spec=self.input_specs,
                opset_version=SerializationSettings.onnx_opset,
            )

            onnx_path = f"{output_filepath}.onnx"
            self._rename_unity_outputs(onnx_path)

        logger.info(f"Exported {onnx_path}")
