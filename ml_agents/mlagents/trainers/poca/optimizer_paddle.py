from typing import Dict, cast, List, Tuple, Optional
from collections import defaultdict
import attr
import numpy as np
import paddle
import paddle.nn as nn
from mlagents.trainers.paddle_entities.components.reward_providers.extrinsic_reward_provider import (
    ExtrinsicRewardProvider,
)
from mlagents.trainers.buffer import (
    AgentBuffer,
    BufferKey,
    RewardSignalUtil,
    AgentBufferField,
)

from mlagents_envs.timers import timed
from mlagents_envs.base_env import ObservationSpec, ActionSpec
from mlagents.trainers.policy.paddle_policy import PaddlePolicy
from mlagents.trainers.optimizer.paddle_optimizer import PaddleOptimizer
from mlagents.trainers.settings import (
    RewardSignalSettings,
    RewardSignalType,
    TrainerSettings,
    NetworkSettings,
    OnPolicyHyperparamSettings,
    ScheduleType,
)
from mlagents.trainers.paddle_entities.networks import Critic, MultiAgentNetworkBody
from mlagents.trainers.paddle_entities.decoders import ValueHeads
from mlagents.trainers.paddle_entities.agent_action import AgentAction
from mlagents.trainers.paddle_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.paddle_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil, GroupObsUtil

from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class POCASettings(OnPolicyHyperparamSettings):
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    beta_schedule: ScheduleType = ScheduleType.LINEAR
    epsilon_schedule: ScheduleType = ScheduleType.LINEAR


class PaddlePOCAOptimizer(PaddleOptimizer):
    class POCAValueNetwork(nn.Layer, Critic):
        """
        The POCAValueNetwork uses the MultiAgentNetworkBody to compute the value
        and POCA baseline for a variable number of agents in a group that all
        share the same observation and action space.
        """

        def __init__(
            self,
            stream_names: List[str],
            observation_specs: List[ObservationSpec],
            network_settings: NetworkSettings,
            action_spec: ActionSpec,
        ):
            nn.Layer.__init__(self)
            self.network_body = MultiAgentNetworkBody(
                observation_specs, network_settings, action_spec
            )
            if network_settings.memory is not None:
                encoding_size = network_settings.memory.memory_size // 2
            else:
                encoding_size = network_settings.hidden_units

            self.value_heads = ValueHeads(stream_names, encoding_size + 1, 1)
            # The + 1 is for the normalized number of agents

        @property
        def memory_size(self) -> int:
            return self.network_body.memory_size

        def update_normalization(self, buffer: AgentBuffer) -> None:
            self.network_body.update_normalization(buffer)

        def baseline(
            self,
            obs_without_actions: List[paddle.Tensor],
            obs_with_actions: Tuple[List[List[paddle.Tensor]], List[AgentAction]],
            memories: Optional[paddle.Tensor] = None,
            sequence_length: int = 1,
        ) -> Tuple[Dict[str, paddle.Tensor], paddle.Tensor]:
            """
            The POCA baseline marginalizes the action of the agent associated with self_obs.
            It calls the forward pass of the MultiAgentNetworkBody with the state action
            pairs of groupmates but just the state of the agent in question.
            """
            (obs, actions) = obs_with_actions
            encoding, memories = self.network_body(
                obs_only=[obs_without_actions],
                obs=obs,
                actions=actions,
                memories=memories,
                sequence_length=sequence_length,
            )

            value_outputs, critic_mem_out = self.forward(
                encoding, memories, sequence_length
            )
            return value_outputs, critic_mem_out

        def critic_pass(
            self,
            obs: List[List[paddle.Tensor]],
            memories: Optional[paddle.Tensor] = None,
            sequence_length: int = 1,
        ) -> Tuple[Dict[str, paddle.Tensor], paddle.Tensor]:
            """
            A centralized value function. It calls the forward pass of MultiAgentNetworkBody
            with just the states of all agents.
            """
            encoding, memories = self.network_body(
                obs_only=obs,
                obs=[],
                actions=[],
                memories=memories,
                sequence_length=sequence_length,
            )

            value_outputs, critic_mem_out = self.forward(
                encoding, memories, sequence_length
            )
            return value_outputs, critic_mem_out

        def forward(
            self,
            encoding: paddle.Tensor,
            memories: Optional[paddle.Tensor] = None,
            sequence_length: int = 1,
        ) -> Tuple[paddle.Tensor, paddle.Tensor]:

            output = self.value_heads(encoding)
            return output, memories

    def __init__(self, policy: PaddlePolicy, trainer_settings: TrainerSettings):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        """
        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        self._critic = PaddlePOCAOptimizer.POCAValueNetwork(
            reward_signal_names,
            policy.behavior_spec.observation_specs,
            network_settings=trainer_settings.network_settings,
            action_spec=policy.behavior_spec.action_spec,
        )

        # In Paddle, explicit .to(device) is rarely needed if global device is set,
        # but kept here for logic consistency with PyTorch version if needed.
        # self._critic.to(paddle.get_device())

        params = list(self.policy.actor.parameters()) + list(self.critic.parameters())

        self.hyperparameters: POCASettings = cast(
            POCASettings, trainer_settings.hyperparameters
        )

        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.epsilon_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.beta_schedule,
            self.hyperparameters.beta,
            1e-5,
            self.trainer_settings.max_steps,
        )

        self.optimizer = paddle.optimizer.Adam(
            learning_rate=self.trainer_settings.hyperparameters.learning_rate,
            parameters=params
        )

        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.stream_names = list(self.reward_signals.keys())
        self.value_memory_dict: Dict[str, paddle.Tensor] = {}
        self.baseline_memory_dict: Dict[str, paddle.Tensor] = {}

    def create_reward_signals(
        self, reward_signal_configs: Dict[RewardSignalType, RewardSignalSettings]
    ) -> None:
        """
        Create reward signals. Override default to provide warnings for Curiosity and
        GAIL, and make sure Extrinsic adds team rewards.
        """
        for reward_signal in reward_signal_configs.keys():
            if reward_signal != RewardSignalType.EXTRINSIC:
                logger.warning(
                    f"Reward signal {reward_signal.value.capitalize()} is not supported with the POCA trainer; "
                    "results may be unexpected."
                )
        super().create_reward_signals(reward_signal_configs)
        # Make sure we add the groupmate rewards in POCA
        for reward_provider in self.reward_signals.values():
            if isinstance(reward_provider, ExtrinsicRewardProvider):
                reward_provider.add_groupmate_rewards = True

    @property
    def critic(self):
        return self._critic

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model.
        """
        # Get decayed parameters
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())
        returns = {}
        old_values = {}
        old_baseline_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)]
            )
            old_baseline_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.baseline_estimates_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        groupmate_obs = GroupObsUtil.from_buffer(batch, n_obs)
        groupmate_obs = [
            [ModelUtils.list_to_tensor(obs) for obs in _groupmate_obs]
            for _groupmate_obs in groupmate_obs
        ]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)
        groupmate_actions = AgentAction.group_from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = paddle.stack(memories).unsqueeze(0)

        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length
            )
        ]

        baseline_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.BASELINE_MEMORY][i])
            for i in range(
                0, len(batch[BufferKey.BASELINE_MEMORY]), self.policy.sequence_length
            )
        ]

        if len(value_memories) > 0:
            value_memories = paddle.stack(value_memories).unsqueeze(0)
            baseline_memories = paddle.stack(baseline_memories).unsqueeze(0)

        run_out = self.policy.actor.get_stats(
            current_obs,
            actions,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )

        log_probs = run_out["log_probs"]
        entropy = run_out["entropy"]

        all_obs = [current_obs] + groupmate_obs
        values, _ = self.critic.critic_pass(
            all_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )
        groupmate_obs_and_actions = (groupmate_obs, groupmate_actions)
        baselines, _ = self.critic.baseline(
            current_obs,
            groupmate_obs_and_actions,
            memories=baseline_memories,
            sequence_length=self.policy.sequence_length,
        )
        old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        log_probs = log_probs.flatten()

        # Paddle equivalent for dtype=torch.bool is dtype='bool' or paddle.bool
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype='bool')

        baseline_loss = ModelUtils.trust_region_value_loss(
            baselines, old_baseline_values, returns, decay_eps, loss_masks
        )
        value_loss = ModelUtils.trust_region_value_loss(
            values, old_values, returns, decay_eps, loss_masks
        )
        policy_loss = ModelUtils.trust_region_policy_loss(
            ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
            log_probs,
            old_log_probs,
            loss_masks,
            decay_eps,
        )

        loss = (
            policy_loss
            + 0.5 * (value_loss + 0.5 * baseline_loss)
            - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
        )

        # Set optimizer learning rate
        self.optimizer.set_lr(decay_lr)
        self.optimizer.clear_grad()
        loss.backward()

        self.optimizer.step()

        update_stats = {
            "Losses/Policy Loss": paddle.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Losses/Baseline Loss": baseline_loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
        }

        return update_stats

    def get_modules(self):
        modules = {"Optimizer:adam": self.optimizer, "Optimizer:critic": self._critic}
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules

    def _evaluate_by_sequence_team(
        self,
        self_obs: List[paddle.Tensor],
        obs: List[List[paddle.Tensor]],
        actions: List[AgentAction],
        init_value_mem: paddle.Tensor,
        init_baseline_mem: paddle.Tensor,
    ) -> Tuple[
        Dict[str, paddle.Tensor],
        Dict[str, paddle.Tensor],
        AgentBufferField,
        AgentBufferField,
        paddle.Tensor,
        paddle.Tensor,
    ]:
        """
        Evaluate a trajectory sequence-by-sequence, assembling the result.
        """
        num_experiences = self_obs[0].shape[0]
        all_next_value_mem = AgentBufferField()
        all_next_baseline_mem = AgentBufferField()

        leftover_seq_len = num_experiences % self.policy.sequence_length

        all_values: Dict[str, List[np.ndarray]] = defaultdict(list)
        all_baseline: Dict[str, List[np.ndarray]] = defaultdict(list)
        _baseline_mem = init_baseline_mem
        _value_mem = init_value_mem

        # Evaluate other trajectories, carrying over _mem after each
        for seq_num in range(num_experiences // self.policy.sequence_length):
            for _ in range(self.policy.sequence_length):
                all_next_value_mem.append(ModelUtils.to_numpy(_value_mem.squeeze()))
                all_next_baseline_mem.append(
                    ModelUtils.to_numpy(_baseline_mem.squeeze())
                )

            start = seq_num * self.policy.sequence_length
            end = (seq_num + 1) * self.policy.sequence_length

            self_seq_obs = []
            groupmate_seq_obs = []
            groupmate_seq_act = []
            seq_obs = []
            for _self_obs in self_obs:
                seq_obs.append(_self_obs[start:end])
            self_seq_obs.append(seq_obs)

            for groupmate_obs, groupmate_action in zip(obs, actions):
                seq_obs = []
                for _obs in groupmate_obs:
                    sliced_seq_obs = _obs[start:end]
                    seq_obs.append(sliced_seq_obs)
                groupmate_seq_obs.append(seq_obs)
                _act = groupmate_action.slice(start, end)
                groupmate_seq_act.append(_act)

            all_seq_obs = self_seq_obs + groupmate_seq_obs
            values, _value_mem = self.critic.critic_pass(
                all_seq_obs, _value_mem, sequence_length=self.policy.sequence_length
            )
            for signal_name, _val in values.items():
                all_values[signal_name].append(_val)

            groupmate_obs_and_actions = (groupmate_seq_obs, groupmate_seq_act)
            baselines, _baseline_mem = self.critic.baseline(
                self_seq_obs[0],
                groupmate_obs_and_actions,
                _baseline_mem,
                sequence_length=self.policy.sequence_length,
            )
            for signal_name, _val in baselines.items():
                all_baseline[signal_name].append(_val)

        # Compute values for the potentially truncated initial sequence
        if leftover_seq_len > 0:
            self_seq_obs = []
            groupmate_seq_obs = []
            groupmate_seq_act = []
            seq_obs = []
            for _self_obs in self_obs:
                last_seq_obs = _self_obs[-leftover_seq_len:]
                seq_obs.append(last_seq_obs)
            self_seq_obs.append(seq_obs)

            for groupmate_obs, groupmate_action in zip(obs, actions):
                seq_obs = []
                for _obs in groupmate_obs:
                    last_seq_obs = _obs[-leftover_seq_len:]
                    seq_obs.append(last_seq_obs)
                groupmate_seq_obs.append(seq_obs)
                _act = groupmate_action.slice(len(_obs) - leftover_seq_len, len(_obs))
                groupmate_seq_act.append(_act)

            seq_obs = []
            for _ in range(leftover_seq_len):
                all_next_value_mem.append(ModelUtils.to_numpy(_value_mem.squeeze()))
                all_next_baseline_mem.append(
                    ModelUtils.to_numpy(_baseline_mem.squeeze())
                )

            all_seq_obs = self_seq_obs + groupmate_seq_obs
            last_values, _value_mem = self.critic.critic_pass(
                all_seq_obs, _value_mem, sequence_length=leftover_seq_len
            )
            for signal_name, _val in last_values.items():
                all_values[signal_name].append(_val)
            groupmate_obs_and_actions = (groupmate_seq_obs, groupmate_seq_act)
            last_baseline, _baseline_mem = self.critic.baseline(
                self_seq_obs[0],
                groupmate_obs_and_actions,
                _baseline_mem,
                sequence_length=leftover_seq_len,
            )
            for signal_name, _val in last_baseline.items():
                all_baseline[signal_name].append(_val)

        # Create one tensor per reward signal
        all_value_tensors = {
            signal_name: paddle.concat(value_list, axis=0)
            for signal_name, value_list in all_values.items()
        }
        all_baseline_tensors = {
            signal_name: paddle.concat(baseline_list, axis=0)
            for signal_name, baseline_list in all_baseline.items()
        }
        next_value_mem = _value_mem
        next_baseline_mem = _baseline_mem
        return (
            all_value_tensors,
            all_baseline_tensors,
            all_next_value_mem,
            all_next_baseline_mem,
            next_value_mem,
            next_baseline_mem,
        )

    def get_trajectory_value_estimates(
        self,
        batch: AgentBuffer,
        next_obs: List[np.ndarray],
        done: bool,
        agent_id: str = "",
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Optional[AgentBufferField]]:
        """
        Override base class method. Unused in the trainer, but needed to make sure class heirarchy is maintained.
        Assume that there are no group obs.
        """
        (
            value_estimates,
            _,
            next_value_estimates,
            all_next_value_mem,
            _,
            _,
        ) = self.get_trajectory_and_baseline_value_estimates(
            batch, next_obs, [], done, agent_id
        )

        return value_estimates, next_value_estimates, all_next_value_mem

    def get_trajectory_and_baseline_value_estimates(
        self,
        batch: AgentBuffer,
        next_obs: List[np.ndarray],
        next_groupmate_obs: List[List[np.ndarray]],
        done: bool,
        agent_id: str = "",
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, float],
        Optional[AgentBufferField],
        Optional[AgentBufferField],
    ]:
        """
        Get value estimates, baseline estimates, and memories for a trajectory, in batch form.
        """

        n_obs = len(self.policy.behavior_spec.observation_specs)

        current_obs = ObsUtil.from_buffer(batch, n_obs)
        groupmate_obs = GroupObsUtil.from_buffer(batch, n_obs)

        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        groupmate_obs = [
            [ModelUtils.list_to_tensor(obs) for obs in _groupmate_obs]
            for _groupmate_obs in groupmate_obs
        ]

        groupmate_actions = AgentAction.group_from_buffer(batch)

        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]
        next_obs = [obs.unsqueeze(0) for obs in next_obs]

        next_groupmate_obs = [
            ModelUtils.list_to_tensor_list(_list_obs)
            for _list_obs in next_groupmate_obs
        ]
        # Expand dimensions of next critic obs
        next_groupmate_obs = [
            [_obs.unsqueeze(0) for _obs in _list_obs]
            for _list_obs in next_groupmate_obs
        ]

        if agent_id in self.value_memory_dict:
            # The agent_id should always be in both since they are added together
            _init_value_mem = self.value_memory_dict[agent_id]
            _init_baseline_mem = self.baseline_memory_dict[agent_id]
        else:
            _init_value_mem = (
                paddle.zeros((1, 1, self.critic.memory_size))
                if self.policy.use_recurrent
                else None
            )
            _init_baseline_mem = (
                paddle.zeros((1, 1, self.critic.memory_size))
                if self.policy.use_recurrent
                else None
            )

        all_obs = (
            [current_obs] + groupmate_obs
            if groupmate_obs is not None
            else [current_obs]
        )
        all_next_value_mem: Optional[AgentBufferField] = None
        all_next_baseline_mem: Optional[AgentBufferField] = None

        with paddle.no_grad():
            if self.policy.use_recurrent:
                (
                    value_estimates,
                    baseline_estimates,
                    all_next_value_mem,
                    all_next_baseline_mem,
                    next_value_mem,
                    next_baseline_mem,
                ) = self._evaluate_by_sequence_team(
                    current_obs,
                    groupmate_obs,
                    groupmate_actions,
                    _init_value_mem,
                    _init_baseline_mem,
                )
            else:
                value_estimates, next_value_mem = self.critic.critic_pass(
                    all_obs, _init_value_mem, sequence_length=batch.num_experiences
                )
                groupmate_obs_and_actions = (groupmate_obs, groupmate_actions)
                baseline_estimates, next_baseline_mem = self.critic.baseline(
                    current_obs,
                    groupmate_obs_and_actions,
                    _init_baseline_mem,
                    sequence_length=batch.num_experiences,
                )

        # Store the memory for the next trajectory
        self.value_memory_dict[agent_id] = next_value_mem
        self.baseline_memory_dict[agent_id] = next_baseline_mem

        all_next_obs = (
            [next_obs] + next_groupmate_obs
            if next_groupmate_obs is not None
            else [next_obs]
        )

        next_value_estimates, _ = self.critic.critic_pass(
            all_next_obs, next_value_mem, sequence_length=1
        )

        for name, estimate in baseline_estimates.items():
            baseline_estimates[name] = ModelUtils.to_numpy(estimate)

        for name, estimate in value_estimates.items():
            value_estimates[name] = ModelUtils.to_numpy(estimate)

        # the base line and V shpuld  not be on the same done flag
        for name, estimate in next_value_estimates.items():
            next_value_estimates[name] = ModelUtils.to_numpy(estimate)

        if done:
            for k in next_value_estimates:
                if not self.reward_signals[k].ignore_done:
                    next_value_estimates[k][-1] = 0.0

        return (
            value_estimates,
            baseline_estimates,
            next_value_estimates,
            all_next_value_mem,
            all_next_baseline_mem,
        )
