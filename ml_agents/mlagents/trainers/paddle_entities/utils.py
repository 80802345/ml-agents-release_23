from typing import List, Optional, Tuple, Dict
import numpy as np
import paddle
import paddle.nn as nn

from mlagents.trainers.paddle_entities.layers import LinearEncoder, Initialization
from mlagents.trainers.paddle_entities.encoders import (
    SimpleVisualEncoder,
    ResNetVisualEncoder,
    NatureVisualEncoder,
    SmallVisualEncoder,
    FullyConnectedVisualEncoder,
    VectorInput,
)
from mlagents.trainers.settings import EncoderType, ScheduleType
from mlagents.trainers.paddle_entities.attention import (
    EntityEmbedding,
    ResidualSelfAttention,
)
from mlagents.trainers.exception import UnityTrainerException
from mlagents_envs.base_env import ObservationSpec, DimensionProperty


class ModelUtils:
    # Minimum supported side for each encoder type. If refactoring an encoder, please
    # adjust these also.
    MIN_RESOLUTION_FOR_ENCODER = {
        EncoderType.FULLY_CONNECTED: 1,
        EncoderType.MATCH3: 5,
        EncoderType.SIMPLE: 20,
        EncoderType.NATURE_CNN: 36,
        EncoderType.RESNET: 15,
    }

    VALID_VISUAL_PROP = frozenset(
        [
            (
                DimensionProperty.NONE,
                DimensionProperty.TRANSLATIONAL_EQUIVARIANCE,
                DimensionProperty.TRANSLATIONAL_EQUIVARIANCE,
            ),
            (DimensionProperty.UNSPECIFIED,) * 3,
        ]
    )

    VALID_VECTOR_PROP = frozenset(
        [(DimensionProperty.NONE,), (DimensionProperty.UNSPECIFIED,)]
    )

    VALID_VAR_LEN_PROP = frozenset(
        [(DimensionProperty.VARIABLE_SIZE, DimensionProperty.NONE)]
    )

    @staticmethod
    def update_learning_rate(optim: paddle.optimizer.Optimizer, lr: float) -> None:
        """
        Apply a learning rate to a paddle optimizer.
        :param optim: Optimizer
        :param lr: Learning rate
        """
        # Paddle Optimizer 的学习率更新方式
        if hasattr(optim, 'set_lr'):
            optim.set_lr(lr)
        else:
            # Fallback assuming 'learning_rate' is a tensor or float on the object
            # Note: Paddle LRScheduler logic is handled internally usually,
            # but for manual update:
            optim._learning_rate = lr
            if isinstance(optim._learning_rate, paddle.Tensor):
                 paddle.assign(paddle.to_tensor(lr, dtype='float32'), optim._learning_rate)

    class DecayedValue:
        def __init__(
            self,
            schedule: ScheduleType,
            initial_value: float,
            min_value: float,
            max_step: int,
        ):
            """
            Object that represnets value of a parameter that should be decayed, assuming it is a function of
            global_step.
            """
            self.schedule = schedule
            self.initial_value = initial_value
            self.min_value = min_value
            self.max_step = max_step

        def get_value(self, global_step: int) -> float:
            """
            Get the value at a given global step.
            :param global_step: Step count.
            :returns: Decayed value at this global step.
            """
            if self.schedule == ScheduleType.CONSTANT:
                return self.initial_value
            elif self.schedule == ScheduleType.LINEAR:
                return ModelUtils.polynomial_decay(
                    self.initial_value, self.min_value, self.max_step, global_step
                )
            else:
                raise UnityTrainerException(f"The schedule {self.schedule} is invalid.")

    @staticmethod
    def polynomial_decay(
        initial_value: float,
        min_value: float,
        max_step: int,
        global_step: int,
        power: float = 1.0,
    ) -> float:
        """
        Get a decayed value based on a polynomial schedule.
        """
        global_step = min(global_step, max_step)
        decayed_value = (initial_value - min_value) * (
            1 - float(global_step) / max_step
        ) ** (power) + min_value
        return decayed_value

    @staticmethod
    def get_encoder_for_type(encoder_type: EncoderType) -> nn.Layer:
        ENCODER_FUNCTION_BY_TYPE = {
            EncoderType.SIMPLE: SimpleVisualEncoder,
            EncoderType.NATURE_CNN: NatureVisualEncoder,
            EncoderType.RESNET: ResNetVisualEncoder,
            EncoderType.MATCH3: SmallVisualEncoder,
            EncoderType.FULLY_CONNECTED: FullyConnectedVisualEncoder,
        }
        return ENCODER_FUNCTION_BY_TYPE.get(encoder_type)

    @staticmethod
    def _check_resolution_for_encoder(
        height: int, width: int, vis_encoder_type: EncoderType
    ) -> None:
        min_res = ModelUtils.MIN_RESOLUTION_FOR_ENCODER[vis_encoder_type]
        if height < min_res or width < min_res:
            raise UnityTrainerException(
                f"Visual observation resolution ({width}x{height}) is too small for"
                f"the provided EncoderType ({vis_encoder_type.value}). The min dimension is {min_res}"
            )

    @staticmethod
    def get_encoder_for_obs(
        obs_spec: ObservationSpec,
        normalize: bool,
        h_size: int,
        attention_embedding_size: int,
        vis_encode_type: EncoderType,
    ) -> Tuple[nn.Layer, int]:
        """
        Returns the encoder and the size of the appropriate encoder.
        """
        shape = obs_spec.shape
        dim_prop = obs_spec.dimension_property

        # VISUAL
        if dim_prop in ModelUtils.VALID_VISUAL_PROP:
            visual_encoder_class = ModelUtils.get_encoder_for_type(vis_encode_type)
            ModelUtils._check_resolution_for_encoder(
                shape[1], shape[2], vis_encode_type
            )
            # Paddle image layers usually expect (C, H, W) or (N, C, H, W)
            # mlagents logic seems to pass (H, W, C) or similar to constructor,
            # ensure the Encoder classes handle it.
            return (visual_encoder_class(shape[1], shape[2], shape[0], h_size), h_size)
        # VECTOR
        if dim_prop in ModelUtils.VALID_VECTOR_PROP:
            return (VectorInput(shape[0], normalize), shape[0])
        # VARIABLE LENGTH
        if dim_prop in ModelUtils.VALID_VAR_LEN_PROP:
            return (
                EntityEmbedding(
                    entity_size=shape[1],
                    entity_num_max_elements=shape[0],
                    embedding_size=attention_embedding_size,
                ),
                0,
            )
        # OTHER
        raise UnityTrainerException(f"Unsupported Sensor with specs {obs_spec}")

    @staticmethod
    def create_input_processors(
        observation_specs: List[ObservationSpec],
        h_size: int,
        vis_encode_type: EncoderType,
        attention_embedding_size: int,
        normalize: bool = False,
    ) -> Tuple[nn.LayerList, List[int]]:
        """
        Creates visual and vector encoders, along with their normalizers.
        """
        encoders: List[nn.Layer] = []
        embedding_sizes: List[int] = []
        for obs_spec in observation_specs:
            encoder, embedding_size = ModelUtils.get_encoder_for_obs(
                obs_spec, normalize, h_size, attention_embedding_size, vis_encode_type
            )
            encoders.append(encoder)
            embedding_sizes.append(embedding_size)

        x_self_size = sum(embedding_sizes)  # The size of the "self" embedding
        if x_self_size > 0:
            for enc in encoders:
                if isinstance(enc, EntityEmbedding):
                    enc.add_self_embedding(attention_embedding_size)
        return (nn.LayerList(encoders), embedding_sizes)

    @staticmethod
    def list_to_tensor(
        ndarray_list: List[np.ndarray], dtype: str = 'float32'
    ) -> paddle.Tensor:
        """
        Converts a list of numpy arrays into a tensor.
        """
        # Paddle automatically handles device placement based on set_device
        return paddle.to_tensor(np.asanyarray(ndarray_list), dtype=dtype)

    @staticmethod
    def list_to_tensor_list(
        ndarray_list: List[np.ndarray], dtype: str = 'float32'
    ) -> List[paddle.Tensor]:
        """
        Converts a list of numpy arrays into a list of tensors.
        """
        return [
            paddle.to_tensor(np.asanyarray(_arr), dtype=dtype)
            for _arr in ndarray_list
        ]

    @staticmethod
    def to_numpy(tensor: paddle.Tensor) -> np.ndarray:
        """
        Converts a Paddle Tensor to a numpy array.
        """
        return tensor.detach().cpu().numpy()

    @staticmethod
    def break_into_branches(
        concatenated_logits: paddle.Tensor, action_size: List[int]
    ) -> List[paddle.Tensor]:
        """
        Takes a concatenated set of logits and breaks it up into one Tensor per branch.
        """
        action_idx = [0] + list(np.cumsum(action_size))
        branched_logits = [
            concatenated_logits[:, action_idx[i] : action_idx[i + 1]]
            for i in range(len(action_size))
        ]
        return branched_logits

    @staticmethod
    def actions_to_onehot(
        discrete_actions: paddle.Tensor, action_size: List[int]
    ) -> List[paddle.Tensor]:
        """
        Takes a tensor of discrete actions and turns it into a List of onehot encoding.
        """
        onehot_branches = [
            # Paddle one_hot requires int input
            paddle.nn.functional.one_hot(_act.cast('int64'), num_classes=action_size[i]).cast('float32')
            for i, _act in enumerate(discrete_actions.cast('int64').t()) # .t() is transpose equivalent for 2D
        ]
        return onehot_branches

    @staticmethod
    def dynamic_partition(
        data: paddle.Tensor, partitions: paddle.Tensor, num_partitions: int
    ) -> List[paddle.Tensor]:
        """
        Splits the data Tensor input into num_partitions Tensors according to the indices in partitions.
        """
        res: List[paddle.Tensor] = []
        for i in range(num_partitions):
            # paddle.nonzero returns [k, d], squeeze(1) makes it [k] indices
            indices = paddle.nonzero(partitions == i).squeeze(1)
            # Paddle supports advanced indexing with tensors
            res += [paddle.gather(data, indices)]
        return res

    @staticmethod
    def masked_mean(tensor: paddle.Tensor, masks: paddle.Tensor) -> paddle.Tensor:
        """
        Returns the mean of the tensor but ignoring the values specified by masks.
        """
        if tensor.ndim == 0:
            return (tensor * masks).sum() / paddle.clip(
                (paddle.ones_like(tensor) * masks).cast('float32').sum(), min=1.0
            )
        else:
            # Equivalent in Paddle: transpose with reversed axes
            perm = list(range(tensor.ndim - 1, -1, -1))

            # Using simple boolean masking math instead of permute if the goal is just global sum?
            # The original code permutes to reverse dimensions before summing.
            # If it's a scalar sum in the end, permute doesn't change the sum value,
            # but it might affect the order of operations for numerical stability or broadcasting?
            # We stick to the translation:

            transposed_tensor = paddle.transpose(tensor, perm)
            transposed_ones = paddle.transpose(paddle.ones_like(tensor), perm)
            # masks = masks.reshape((-1, 1))
            # 新增：将布尔掩码转为float32类型（与transposed_tensor一致）
            masks_float = masks.astype(paddle.float32)
            numerator = (transposed_tensor * masks_float).sum()
            denominator = (transposed_ones * masks_float).cast('float32').sum()

            return numerator / paddle.clip(denominator, min=1.0)

    @staticmethod
    def soft_update(source: nn.Layer, target: nn.Layer, tau: float) -> None:
        """
        Performs an in-place polyak update of the target module based on the source.
        target = tau * source + (1-tau) * target
        """
        with paddle.no_grad():
            for source_param, target_param in zip(
                source.parameters(), target.parameters()
            ):
                # Paddle Tensor in-place operations
                # target = target * (1-tau)
                target_param.scale_(1.0 - tau)
                # target = target + source * tau
                target_param.add_(source_param * tau)

    @staticmethod
    def create_residual_self_attention(
        input_processors: nn.LayerList, embedding_sizes: List[int], hidden_size: int
    ) -> Tuple[Optional[ResidualSelfAttention], Optional[LinearEncoder]]:
        """
        Creates an RSA if there are variable length observations found in the input processors.
        """
        rsa, x_self_encoder = None, None
        entity_num_max: int = 0
        var_processors = [p for p in input_processors if isinstance(p, EntityEmbedding)]
        for processor in var_processors:
            entity_max: int = processor.entity_num_max_elements
            # Only adds entity max if it was known at construction
            if entity_max > 0:
                entity_num_max += entity_max
        if len(var_processors) > 0:
            if sum(embedding_sizes):
                x_self_encoder = LinearEncoder(
                    sum(embedding_sizes),
                    1,
                    hidden_size,
                    kernel_init=Initialization.Normal,
                    kernel_gain=(0.125 / hidden_size) ** 0.5,
                )
            rsa = ResidualSelfAttention(hidden_size, entity_num_max)
        return rsa, x_self_encoder

    @staticmethod
    def trust_region_value_loss(
        values: Dict[str, paddle.Tensor],
        old_values: Dict[str, paddle.Tensor],
        returns: Dict[str, paddle.Tensor],
        epsilon: float,
        loss_masks: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Evaluates value loss, clipping to stay within a trust region of old value estimates.
        """
        value_losses = []
        for name, head in values.items():
            old_val_tensor = old_values[name]
            returns_tensor = returns[name]
            clipped_value_estimate = old_val_tensor + paddle.clip(
                head - old_val_tensor, -1 * epsilon, epsilon
            )
            v_opt_a = (returns_tensor - head) ** 2
            v_opt_b = (returns_tensor - clipped_value_estimate) ** 2
            value_loss = ModelUtils.masked_mean(paddle.maximum(v_opt_a, v_opt_b), loss_masks)
            value_losses.append(value_loss)
        value_loss = paddle.mean(paddle.stack(value_losses))
        return value_loss

    @staticmethod
    def trust_region_policy_loss(
        advantages: paddle.Tensor,
        log_probs: paddle.Tensor,
        old_log_probs: paddle.Tensor,
        loss_masks: paddle.Tensor,
        epsilon: float,
    ) -> paddle.Tensor:
        """
        Evaluate policy loss clipped to stay within a trust region. Used for PPO and POCA.
        """
        advantage = advantages.unsqueeze(-1)
        r_theta = paddle.exp(log_probs - old_log_probs)
        p_opt_a = r_theta * advantage
        p_opt_b = paddle.clip(r_theta, 1.0 - epsilon, 1.0 + epsilon) * advantage
        policy_loss = -1 * ModelUtils.masked_mean(
            paddle.minimum(p_opt_a, p_opt_b), loss_masks
        )
        return policy_loss
