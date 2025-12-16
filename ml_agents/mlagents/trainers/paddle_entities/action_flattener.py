from typing import List
import paddle

from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.paddle_entities.agent_action import AgentAction
from mlagents.trainers.paddle_entities.utils import ModelUtils


class ActionFlattener:
    def __init__(self, action_spec: ActionSpec):
        """
        A helper class that creates the flattened form of an AgentAction object.
        The flattened form is the continuous action concatenated with the
        concatenated one hot encodings of the discrete actions.
        :param action_spec: An ActionSpec that describes the action space dimensions
        """
        self._specs = action_spec

    @property
    def flattened_size(self) -> int:
        """
        The flattened size is the continuous size plus the sum of the branch sizes
        since discrete actions are encoded as one hots.
        """
        return self._specs.continuous_size + sum(self._specs.discrete_branches)

    def forward(self, action: AgentAction) -> paddle.Tensor:
        """
        Returns a tensor corresponding the flattened action
        :param action: An AgentAction object
        """
        action_list: List[paddle.Tensor] = []
        if self._specs.continuous_size > 0:
            action_list.append(action.continuous_tensor)

        if self._specs.discrete_size > 0:
            discrete_tensor_int = action.discrete_tensor.cast('int64')

            flat_discrete = paddle.concat(
                ModelUtils.actions_to_onehot(
                    discrete_tensor_int,
                    self._specs.discrete_branches,
                ),
                axis=1,
            )
            action_list.append(flat_discrete)
        if not action_list:
            return paddle.to_tensor([], dtype='float32')

        return paddle.concat(action_list, axis=1)
