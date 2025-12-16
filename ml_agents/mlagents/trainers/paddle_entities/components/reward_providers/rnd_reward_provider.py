import numpy as np
from typing import Dict
import paddle
import paddle.nn as nn

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.paddle_entities.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)
from mlagents.trainers.settings import RNDSettings

from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs import logging_util
from mlagents.trainers.paddle_entities.utils import ModelUtils
from mlagents.trainers.paddle_entities.networks import NetworkBody
from mlagents.trainers.trajectory import ObsUtil

logger = logging_util.get_logger(__name__)


class RNDRewardProvider(BaseRewardProvider):
    """
    Implementation of Random Network Distillation : https://arxiv.org/pdf/1810.12894.pdf
    """

    def __init__(self, specs: BehaviorSpec, settings: RNDSettings) -> None:
        super().__init__(specs, settings)
        self._ignore_done = True
        self._random_network = RNDNetwork(specs, settings)
        self._training_network = RNDNetwork(specs, settings)

        # Paddle handles device placement globally
        # self._random_network.to(default_device())
        # self._training_network.to(default_device())

        self.optimizer = paddle.optimizer.Adam(
            learning_rate=settings.learning_rate,
            parameters=self._training_network.parameters()
        )

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        with paddle.no_grad():
            target = self._random_network(mini_batch)
            prediction = self._training_network(mini_batch)
            # torch.sum(..., dim=1) -> paddle.sum(..., axis=1)
            rewards = paddle.sum((prediction - target) ** 2, axis=1)
        return ModelUtils.to_numpy(rewards)

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:
        # Target network inference (fixed)
        with paddle.no_grad():
            target = self._random_network(mini_batch)

        # Training network forward
        prediction = self._training_network(mini_batch)

        loss = paddle.mean(paddle.sum((prediction - target) ** 2, axis=1))

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()

        return {"Losses/RND Loss": float(loss.item())}

    def get_modules(self):
        return {
            f"Module:{self.name}-pred": self._training_network,
            f"Module:{self.name}-target": self._random_network,
        }


class RNDNetwork(nn.Layer):
    EPSILON = 1e-10

    def __init__(self, specs: BehaviorSpec, settings: RNDSettings) -> None:
        super().__init__()
        state_encoder_settings = settings.network_settings
        if state_encoder_settings.memory is not None:
            state_encoder_settings.memory = None
            logger.warning(
                "memory was specified in network_settings but is not supported by RND. It is being ignored."
            )

        self._encoder = NetworkBody(specs.observation_specs, state_encoder_settings)

    def forward(self, mini_batch: AgentBuffer) -> paddle.Tensor:
        n_obs = len(self._encoder.processors)
        np_obs = ObsUtil.from_buffer(mini_batch, n_obs)
        # Convert to tensors
        tensor_obs = [ModelUtils.list_to_tensor(obs) for obs in np_obs]

        hidden, _ = self._encoder.forward(tensor_obs)
        self._encoder.update_normalization(mini_batch)
        return hidden
