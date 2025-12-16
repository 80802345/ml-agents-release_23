from typing import Optional, Dict, List
import numpy as np
import paddle
import paddle.nn as nn

from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.paddle_entities.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)
from mlagents.trainers.settings import GAILSettings
from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs import logging_util

from mlagents.trainers.paddle_entities.utils import ModelUtils
from mlagents.trainers.paddle_entities.agent_action import AgentAction
from mlagents.trainers.paddle_entities.action_flattener import ActionFlattener
from mlagents.trainers.paddle_entities.networks import NetworkBody
from mlagents.trainers.paddle_entities.layers import linear_layer, Initialization
from mlagents.trainers.demo_loader import demo_to_buffer
from mlagents.trainers.trajectory import ObsUtil

logger = logging_util.get_logger(__name__)


class GAILRewardProvider(BaseRewardProvider):
    def __init__(self, specs: BehaviorSpec, settings: GAILSettings) -> None:
        super().__init__(specs, settings)
        self._ignore_done = False
        self._discriminator_network = DiscriminatorNetwork(specs, settings)
        # Paddle handles device placement globally
        # self._discriminator_network.to(default_device())

        _, self._demo_buffer = demo_to_buffer(
            settings.demo_path, 1, specs
        )  # This is supposed to be the sequence length but we do not have access here

        self.optimizer = paddle.optimizer.Adam(
            learning_rate=settings.learning_rate,
            parameters=self._discriminator_network.parameters()
        )

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        with paddle.no_grad():
            estimates, _ = self._discriminator_network.compute_estimate(
                mini_batch, use_vail_noise=False
            )
            # paddle.squeeze dim -> axis
            return ModelUtils.to_numpy(
                -paddle.log(
                    1.0
                    - estimates.squeeze(axis=1)
                    * (1.0 - self._discriminator_network.EPSILON)
                )
            )

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:

        expert_batch = self._demo_buffer.sample_mini_batch(
            mini_batch.num_experiences, 1
        )
        self._discriminator_network.encoder.update_normalization(expert_batch)

        loss, stats_dict = self._discriminator_network.compute_loss(
            mini_batch, expert_batch
        )

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()

        return stats_dict

    def get_modules(self):
        return {f"Module:{self.name}": self._discriminator_network}


class DiscriminatorNetwork(nn.Layer):
    gradient_penalty_weight = 10.0
    z_size = 128
    alpha = 0.0005
    mutual_information = 0.5
    EPSILON = 1e-7
    initial_beta = 0.0

    def __init__(self, specs: BehaviorSpec, settings: GAILSettings) -> None:
        super().__init__()
        self._use_vail = settings.use_vail
        self._settings = settings

        encoder_settings = settings.network_settings
        if encoder_settings.memory is not None:
            encoder_settings.memory = None
            logger.warning(
                "memory was specified in network_settings but is not supported by GAIL. It is being ignored."
            )

        self._action_flattener = ActionFlattener(specs.action_spec)
        unencoded_size = (
            self._action_flattener.flattened_size + 1 if settings.use_actions else 0
        )  # +1 is for dones
        self.encoder = NetworkBody(
            specs.observation_specs, encoder_settings, unencoded_size
        )

        estimator_input_size = encoder_settings.hidden_units
        if settings.use_vail:
            estimator_input_size = self.z_size
            # torch.nn.Parameter -> self.create_parameter
            self._z_sigma = self.create_parameter(
                shape=[self.z_size],
                default_initializer=nn.initializer.Constant(1.0),
                is_bias=False
            ) # requires_grad=True by default

            self._z_mu_layer = linear_layer(
                encoder_settings.hidden_units,
                self.z_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=0.1,
            )

            # requires_grad=False -> stop_gradient=True
            self._beta = self.create_parameter(
                shape=[1],
                default_initializer=nn.initializer.Constant(self.initial_beta),
                is_bias=False
            )
            self._beta.stop_gradient = True

        self._estimator = nn.Sequential(
            linear_layer(estimator_input_size, 1, kernel_gain=0.2), nn.Sigmoid()
        )

    def get_action_input(self, mini_batch: AgentBuffer) -> paddle.Tensor:
        """
        Creates the action Tensor.
        """
        return self._action_flattener.forward(AgentAction.from_buffer(mini_batch))

    def get_state_inputs(self, mini_batch: AgentBuffer) -> List[paddle.Tensor]:
        """
        Creates the observation input.
        """
        n_obs = len(self.encoder.processors)
        np_obs = ObsUtil.from_buffer(mini_batch, n_obs)
        tensor_obs = [ModelUtils.list_to_tensor(obs) for obs in np_obs]
        return tensor_obs

    def compute_estimate(
        self, mini_batch: AgentBuffer, use_vail_noise: bool = False
    ) -> paddle.Tensor:
        """
        Given a mini_batch, computes the estimate.
        """
        inputs = self.get_state_inputs(mini_batch)
        if self._settings.use_actions:
            actions = self.get_action_input(mini_batch)
            # done buffer usually float/int
            dones = ModelUtils.list_to_tensor(
                mini_batch[BufferKey.DONE], dtype='float32'
            ).unsqueeze(1)

            action_inputs = paddle.concat([actions, dones], axis=1)
            hidden, _ = self.encoder(inputs, action_inputs)
        else:
            hidden, _ = self.encoder(inputs)

        z_mu: Optional[paddle.Tensor] = None
        if self._settings.use_vail:
            z_mu = self._z_mu_layer(hidden)
            # paddle.randn_like or paddle.randn(shape)
            noise = paddle.randn(z_mu.shape)
            hidden = z_mu + noise * self._z_sigma * float(use_vail_noise)

        estimate = self._estimator(hidden)
        return estimate, z_mu

    def compute_loss(
        self, policy_batch: AgentBuffer, expert_batch: AgentBuffer
    ) -> paddle.Tensor:
        """
        Given a policy mini_batch and an expert mini_batch, computes the loss of the discriminator.
        """
        total_loss = paddle.zeros([1])
        stats_dict: Dict[str, np.ndarray] = {}

        policy_estimate, policy_mu = self.compute_estimate(
            policy_batch, use_vail_noise=True
        )
        expert_estimate, expert_mu = self.compute_estimate(
            expert_batch, use_vail_noise=True
        )

        stats_dict["Policy/GAIL Policy Estimate"] = policy_estimate.mean().item()
        stats_dict["Policy/GAIL Expert Estimate"] = expert_estimate.mean().item()

        discriminator_loss = -(
            paddle.log(expert_estimate + self.EPSILON)
            + paddle.log(1.0 - policy_estimate + self.EPSILON)
        ).mean()

        stats_dict["Losses/GAIL Loss"] = discriminator_loss.item()
        total_loss += discriminator_loss

        if self._settings.use_vail:
            # KL divergence loss
            kl_loss = paddle.mean(
                -paddle.sum(
                    1
                    + paddle.log(self._z_sigma**2)
                    - 0.5 * expert_mu**2
                    - 0.5 * policy_mu**2
                    - (self._z_sigma**2),
                    axis=1,
                )
            )
            vail_loss = self._beta * (kl_loss - self.mutual_information)

            # Manual update of non-trainable parameter _beta
            with paddle.no_grad():
                beta_new = paddle.maximum(
                    self._beta + self.alpha * (kl_loss - self.mutual_information),
                    paddle.to_tensor(0.0)
                )
                paddle.assign(beta_new, self._beta)

            total_loss += vail_loss
            stats_dict["Policy/GAIL Beta"] = self._beta.item()
            stats_dict["Losses/GAIL KL Loss"] = kl_loss.item()

        if self.gradient_penalty_weight > 0.0:
            gradient_magnitude_loss = (
                self.gradient_penalty_weight
                * self.compute_gradient_magnitude(policy_batch, expert_batch)
            )
            stats_dict["Policy/GAIL Grad Mag Loss"] = gradient_magnitude_loss.item()
            total_loss += gradient_magnitude_loss

        return total_loss, stats_dict

    def compute_gradient_magnitude(
        self, policy_batch: AgentBuffer, expert_batch: AgentBuffer
    ) -> paddle.Tensor:
        """
        Gradient penalty. Compute gradients w.r.t randomly interpolated input.
        """
        policy_inputs = self.get_state_inputs(policy_batch)
        expert_inputs = self.get_state_inputs(expert_batch)
        interp_inputs = []

        for policy_input, expert_input in zip(policy_inputs, expert_inputs):
            obs_epsilon = paddle.rand(policy_input.shape)
            interp_input = obs_epsilon * policy_input + (1 - obs_epsilon) * expert_input
            interp_input.stop_gradient = False  # Enable gradient calculation
            interp_inputs.append(interp_input)

        if self._settings.use_actions:
            policy_action = self.get_action_input(policy_batch)
            expert_action = self.get_action_input(expert_batch)
            action_epsilon = paddle.rand(policy_action.shape)

            policy_dones = ModelUtils.list_to_tensor(
                policy_batch[BufferKey.DONE], dtype='float32'
            ).unsqueeze(1)
            expert_dones = ModelUtils.list_to_tensor(
                expert_batch[BufferKey.DONE], dtype='float32'
            ).unsqueeze(1)
            dones_epsilon = paddle.rand(policy_dones.shape)

            action_inputs = paddle.concat(
                [
                    action_epsilon * policy_action
                    + (1 - action_epsilon) * expert_action,
                    dones_epsilon * policy_dones + (1 - dones_epsilon) * expert_dones,
                ],
                axis=1,
            )
            action_inputs.stop_gradient = False
            hidden, _ = self.encoder(interp_inputs, action_inputs)
            # Tuple for grad inputs
            encoder_input = interp_inputs + [action_inputs]
        else:
            hidden, _ = self.encoder(interp_inputs)
            encoder_input = interp_inputs

        if self._settings.use_vail:
            use_vail_noise = True
            z_mu = self._z_mu_layer(hidden)
            noise = paddle.randn(z_mu.shape)
            hidden = z_mu + noise * self._z_sigma * float(use_vail_noise)

        estimate = self._estimator(hidden).squeeze(1).sum()

        # Calculate gradients
        # paddle.grad returns a list of gradients corresponding to inputs
        gradients = paddle.grad(
            outputs=[estimate],
            inputs=encoder_input,
            create_graph=True,
            retain_graph=True
        )

        # Norm's gradient could be NaN at 0. Use our own safe_norm
        # We need to aggregate gradients from all inputs (obs + actions)?
        # Typically gradient penalty is on the input space magnitude.
        # If multiple inputs, we sum their norms or concat?
        # The original code takes gradients[0] implying it might only look at the first input
        # OR `encoder_input` was passed as a tuple, so `torch.autograd.grad` returns tuple.
        # Original: gradient = torch.autograd.grad(..., encoder_input, ...)[0]
        # Wait, if encoder_input is a tuple of multiple tensors (e.g. visual + vector + action),
        # grabbing [0] only penalizes the first observation?
        # If the original code did that, we follow it.
        # However, ML-Agents usually has a specific structure.
        # If `interp_inputs` is a list, `encoder_input` is a list.
        # Let's verify `torch.autograd.grad` behavior: returns tuple of gradients matching inputs.

        # If we follow the original code strictly:
        gradient = gradients[0]

        safe_norm = (paddle.sum(gradient**2, axis=1) + self.EPSILON).sqrt()
        gradient_mag = paddle.mean((safe_norm - 1) ** 2)
        return gradient_mag
