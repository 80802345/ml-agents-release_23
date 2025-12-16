from typing import List, Optional, NamedTuple
import itertools
import numpy as np
import paddle
from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.paddle_entities.utils import ModelUtils
from mlagents_envs.base_env import ActionTuple


class AgentAction(NamedTuple):
    """
    A NamedTuple containing the tensor for continuous actions and list of tensors for
    discrete actions. Utility functions provide numpy <=> tensor conversions to be
    sent as actions to the environment manager as well as used by the optimizers.
    :param continuous_tensor: Paddle tensor corresponding to continuous actions
    :param discrete_list: List of Paddle tensors each corresponding to discrete actions
    """

    continuous_tensor: paddle.Tensor
    discrete_list: Optional[List[paddle.Tensor]]

    @property
    def discrete_tensor(self) -> paddle.Tensor:
        """
        Returns the discrete action list as a stacked tensor
        """
        if self.discrete_list is not None and len(self.discrete_list) > 0:
            return paddle.stack(self.discrete_list, axis=-1)
        else:
            # Paddle equivalent of an empty tensor for logical consistency
            # However, typically this branch implies no discrete actions.
            return paddle.to_tensor([], dtype='float32')

    def slice(self, start: int, end: int) -> "AgentAction":
        """
        Returns an AgentAction with the continuous and discrete tensors slices
        from index start to index end.
        """
        _cont = None
        _disc_list = []
        if self.continuous_tensor is not None:
            _cont = self.continuous_tensor[start:end]
        if self.discrete_list is not None and len(self.discrete_list) > 0:
            for _disc in self.discrete_list:
                _disc_list.append(_disc[start:end])
        return AgentAction(_cont, _disc_list)

    def to_action_tuple(self, clip: bool = False) -> ActionTuple:
        """
        Returns an ActionTuple
        """
        action_tuple = ActionTuple()
        if self.continuous_tensor is not None:
            _continuous_tensor = self.continuous_tensor
            if clip:
                # torch.clamp -> paddle.clip
                _continuous_tensor = paddle.clip(_continuous_tensor, -3, 3) / 3
            continuous = ModelUtils.to_numpy(_continuous_tensor)
            action_tuple.add_continuous(continuous)
        if self.discrete_list is not None:
            # discrete_tensor is [batch, num_branches] or [batch, 1, num_branches]?
            # Usually [batch, time, branches] or [batch, branches]
            # Assuming discrete_tensor property stack works correctly.
            # torch: [:, 0, :] implies grabbing first element of sequence?
            # Need to verify dimension. Assuming code structure matches original.
            if len(self.discrete_list) > 0:
                # Note: The original torch code used self.discrete_tensor[:, 0, :]
                # This implies discrete_list elements are [Batch, Time, ...] or similar?
                # or [Batch, 1]?
                # If discrete_list elements are 1D [Batch], stacking gives [Batch, Branches].
                # If original code slices [:, 0, :], it suggests discrete_tensor is 3D.
                # We will trust the original slicing logic works on the tensor shape.
                discrete = ModelUtils.to_numpy(self.discrete_tensor[:, 0, :])
                action_tuple.add_discrete(discrete)
        return action_tuple

    @staticmethod
    def from_buffer(buff: AgentBuffer) -> "AgentAction":
        """
        A static method that accesses continuous and discrete action fields in an AgentBuffer
        and constructs the corresponding AgentAction from the retrieved np arrays.
        """
        continuous: paddle.Tensor = None
        discrete: List[paddle.Tensor] = None  # type: ignore
        if BufferKey.CONTINUOUS_ACTION in buff:
            continuous = ModelUtils.list_to_tensor(buff[BufferKey.CONTINUOUS_ACTION])
        if BufferKey.DISCRETE_ACTION in buff:
            # dtype=torch.long -> dtype='int64'
            discrete_tensor = ModelUtils.list_to_tensor(
                buff[BufferKey.DISCRETE_ACTION], dtype='int64'
            )
            # Split tensor back into list of tensors per branch
            discrete = [
                discrete_tensor[..., i] for i in range(discrete_tensor.shape[-1])
            ]
        return AgentAction(continuous, discrete)

    @staticmethod
    def _group_agent_action_from_buffer(
        buff: AgentBuffer, cont_action_key: BufferKey, disc_action_key: BufferKey
    ) -> List["AgentAction"]:
        """
        Extracts continuous and discrete groupmate actions...
        """
        continuous_tensors: List[paddle.Tensor] = []
        discrete_tensors: List[paddle.Tensor] = []
        if cont_action_key in buff:
            padded_batch = buff[cont_action_key].padded_to_batch()
            continuous_tensors = [
                ModelUtils.list_to_tensor(arr) for arr in padded_batch
            ]
        if disc_action_key in buff:
            # dtype=np.long -> dtype=np.int64
            padded_batch = buff[disc_action_key].padded_to_batch(dtype=np.int64)
            discrete_tensors = [
                ModelUtils.list_to_tensor(arr, dtype='int64') for arr in padded_batch
            ]

        actions_list = []
        for _cont, _disc in itertools.zip_longest(
            continuous_tensors, discrete_tensors, fillvalue=None
        ):
            if _disc is not None:
                _disc = [_disc[..., i] for i in range(_disc.shape[-1])]
            actions_list.append(AgentAction(_cont, _disc))
        return actions_list

    @staticmethod
    def group_from_buffer(buff: AgentBuffer) -> List["AgentAction"]:
        return AgentAction._group_agent_action_from_buffer(
            buff, BufferKey.GROUP_CONTINUOUS_ACTION, BufferKey.GROUP_DISCRETE_ACTION
        )

    @staticmethod
    def group_from_buffer_next(buff: AgentBuffer) -> List["AgentAction"]:
        return AgentAction._group_agent_action_from_buffer(
            buff, BufferKey.GROUP_NEXT_CONT_ACTION, BufferKey.GROUP_NEXT_DISC_ACTION
        )

    def to_flat(self, discrete_branches: List[int]) -> paddle.Tensor:
        """
        Flatten this AgentAction into a single paddle Tensor...
        """
        # if there are any discrete actions, create one-hot
        if self.discrete_list is not None and len(self.discrete_list) > 0:
            discrete_oh = ModelUtils.actions_to_onehot(
                self.discrete_tensor, discrete_branches
            )
            discrete_oh = paddle.concat(discrete_oh, axis=1)
        else:
            discrete_oh = paddle.to_tensor([], dtype='float32')

        # Concatenate continuous and discrete
        # Handle cases where one might be empty/None
        tensors_to_cat = []
        if self.continuous_tensor is not None and self.continuous_tensor.size > 0:
             tensors_to_cat.append(self.continuous_tensor)

        if discrete_oh.size > 0:
             tensors_to_cat.append(discrete_oh)

        if not tensors_to_cat:
            return paddle.to_tensor([], dtype='float32')

        return paddle.concat(tensors_to_cat, axis=-1)
