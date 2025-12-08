from typing import Dict, cast
import attr
import paddle

# Ensure these point to your converted Paddle modules
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents.trainers.policy.paddle_policy import PaddlePolicy
from mlagents.trainers.optimizer.paddle_optimizer import PaddleOptimizer
from mlagents.trainers.settings import (
    TrainerSettings,
    OnPolicyHyperparamSettings,
    ScheduleType,
)
from mlagents.trainers.paddle_entities.networks import ValueNetwork
from mlagents.trainers.paddle_entities.agent_action import AgentAction
from mlagents.trainers.paddle_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.paddle_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil
from mlagents_envs.timers import timed


@attr.s(auto_attribs=True)
class PPOSettings(OnPolicyHyperparamSettings):
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    shared_critic: bool = False
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    beta_schedule: ScheduleType = ScheduleType.LINEAR
    epsilon_schedule: ScheduleType = ScheduleType.LINEAR


class PaddlePPOOptimizer(PaddleOptimizer):
    def __init__(self, policy: PaddlePolicy, trainer_settings: TrainerSettings):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        The PPO optimizer has a value estimator and a loss function.
        :param policy: A PaddlePolicy object that will be updated by this PPO Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the
        properties of the trainer.
        """
        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        self.hyperparameters: PPOSettings = cast(
            PPOSettings, trainer_settings.hyperparameters
        )

        # Paddle parameters list
        params = list(self.policy.actor.parameters())
        if self.hyperparameters.shared_critic:
            self._critic = policy.actor
        else:
            self._critic = ValueNetwork(
                reward_signal_names,
                policy.behavior_spec.observation_specs,
                network_settings=trainer_settings.network_settings,
            )
            # Paddle handles device automatically
            params += list(self._critic.parameters())

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

    @property
    def critic(self):
        return self._critic

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        # Get decayed parameters
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())
        
        returns = {}
        old_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            # torch.stack -> paddle.stack, unsqueeze(0) -> unsqueeze(axis=0)
            memories = paddle.stack(memories).unsqueeze(axis=0)
        else:
            memories = None 

        # Get value memories
        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length
            )
        ]
        if len(value_memories) > 0:
            value_memories = paddle.stack(value_memories).unsqueeze(axis=0)
        else:
            value_memories = None

        run_out = self.policy.actor.get_stats(
            current_obs,
            actions,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )

        log_probs = run_out["log_probs"]
        entropy = run_out["entropy"]

        values, _ = self.critic.critic_pass(
            current_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )
        
        old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        log_probs = log_probs.flatten()
        
        # dtype=torch.bool -> dtype='bool'
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype='bool')
        
        # PPO Loss Calculations (Delegated to ModelUtils which should be Paddle-compatible now)
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
            + 0.5 * value_loss
            - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
        )

        # Set optimizer learning rate
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        
        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        
        update_stats = {
            "Losses/Policy Loss": paddle.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
        }

        return update_stats

    def get_modules(self):
        modules = {
            "Optimizer:value_optimizer": self.optimizer,
            "Optimizer:critic": self._critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules