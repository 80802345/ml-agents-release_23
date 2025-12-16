import abc
from typing import List
import numpy as np
import math
import paddle
import paddle.nn as nn
import paddle.distribution as p_dist
from mlagents.trainers.paddle_entities.layers import linear_layer, Initialization

EPSILON = 1e-7  # Small value to avoid divide by zero


class DistInstance(nn.Layer, abc.ABC):
    @abc.abstractmethod
    def sample(self) -> paddle.Tensor:
        """
        Return a sample from this distribution.
        """
        pass

    @abc.abstractmethod
    def deterministic_sample(self) -> paddle.Tensor:
        """
        Return the most probable sample from this distribution.
        """
        pass

    @abc.abstractmethod
    def log_prob(self, value: paddle.Tensor) -> paddle.Tensor:
        """
        Returns the log probabilities of a particular value.
        :param value: A value sampled from the distribution.
        :returns: Log probabilities of the given value.
        """
        pass

    @abc.abstractmethod
    def entropy(self) -> paddle.Tensor:
        """
        Returns the entropy of this distribution.
        """
        pass

    @abc.abstractmethod
    def exported_model_output(self) -> paddle.Tensor:
        """
        Returns the tensor to be exported to ONNX for the distribution
        """
        pass


class DiscreteDistInstance(DistInstance):
    @abc.abstractmethod
    def all_log_prob(self) -> paddle.Tensor:
        """
        Returns the log probabilities of all actions represented by this distribution.
        """
        pass


class GaussianDistInstance(DistInstance):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def sample(self):
        sample = self.mean + paddle.randn(self.mean.shape) * self.std
        return sample

    def deterministic_sample(self):
        return self.mean

    def log_prob(self, value):
        var = self.std**2
        log_scale = paddle.log(self.std + EPSILON)
        return (
            -((value - self.mean) ** 2) / (2 * var + EPSILON)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )

    def pdf(self, value):
        log_prob = self.log_prob(value)
        return paddle.exp(log_prob)

    def entropy(self):
        return paddle.mean(
            0.5 * paddle.log(2 * math.pi * math.e * self.std**2 + EPSILON),
            axis=1,
            keepdim=True,
        )  # Use equivalent behavior to TF

    def exported_model_output(self):
        return self.sample()


class TanhGaussianDistInstance(GaussianDistInstance):
    def __init__(self, mean, std):
        super().__init__(mean, std)
        self.transform = p_dist.TanhTransform()

    def sample(self):
        unsquashed_sample = super().sample()
        squashed = self.transform.forward(unsquashed_sample)
        return squashed

    def _inverse_tanh(self, value):
        capped_value = paddle.clip(value, -1 + EPSILON, 1 - EPSILON)
        return 0.5 * paddle.log((1 + capped_value) / (1 - capped_value) + EPSILON)

    def log_prob(self, value):
        unsquashed = self.transform.inverse(value)

        # Paddle: transform.forward_log_det_jacobian(x)
        # Formula: p(y) = p(x) / |det| -> log p(y) = log p(x) - log |det|
        log_det = self.transform.forward_log_det_jacobian(unsquashed)

        return super().log_prob(unsquashed) - log_det


class CategoricalDistInstance(DiscreteDistInstance):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        # axis=-1 is default
        self.probs = paddle.nn.functional.softmax(self.logits, axis=-1)

    def sample(self):
        return paddle.multinomial(self.probs, num_samples=1)

    def deterministic_sample(self):
        return paddle.argmax(self.probs, axis=1, keepdim=True)

    def pdf(self, value):

        idx=paddle.arange(start=0,end=len(value),dtype=paddle.int64).unsqueeze(-1)

        probs_perm = self.probs.transpose((1, 0))  
        value_flat = value.flatten().cast('int64')
        probs_selected = probs_perm[value_flat]
        probs_gathered = paddle.take_along_axis(probs_selected, indices=idx, axis=-1)

        result = probs_gathered.squeeze(-1)
        return result


    def log_prob(self, value):
        return paddle.log(self.pdf(value) + EPSILON)

    def all_log_prob(self):
        return paddle.log(self.probs + EPSILON)

    def entropy(self):
        return -paddle.sum(
            self.probs * paddle.log(self.probs + EPSILON), axis=-1
        ).unsqueeze(-1)

    def exported_model_output(self):
        return self.sample()


class GaussianDistribution(nn.Layer):
    def __init__(
        self,
        hidden_size: int,
        num_outputs: int,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__()
        self.conditional_sigma = conditional_sigma
        self.mu = linear_layer(
            hidden_size,
            num_outputs,
            kernel_init=Initialization.KaimingHeNormal,
            kernel_gain=0.2,
            bias_init=Initialization.Zero,
        )
        self.tanh_squash = tanh_squash
        if conditional_sigma:
            self.log_sigma = linear_layer(
                hidden_size,
                num_outputs,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=0.2,
                bias_init=Initialization.Zero,
            )
        else:
            self.log_sigma = self.create_parameter(
                shape=[1, num_outputs],
                default_initializer=nn.initializer.Constant(0.0),
                is_bias=False
            )

    def forward(self, inputs: paddle.Tensor) -> List[DistInstance]:
        mu = self.mu(inputs)
        if self.conditional_sigma:
            log_sigma = paddle.clip(self.log_sigma(inputs), min=-20, max=2)
        else:
            # Expand so that entropy matches batch size.
            log_sigma = mu * 0 + self.log_sigma

        if self.tanh_squash:
            return TanhGaussianDistInstance(mu, paddle.exp(log_sigma))
        else:
            return GaussianDistInstance(mu, paddle.exp(log_sigma))


class MultiCategoricalDistribution(nn.Layer):
    def __init__(self, hidden_size: int, act_sizes: List[int]):
        super().__init__()
        self.act_sizes = act_sizes
        self.branches = self._create_policy_branches(hidden_size)

    def _create_policy_branches(self, hidden_size: int) -> nn.LayerList:
        branches = []
        for size in self.act_sizes:
            branch_output_layer = linear_layer(
                hidden_size,
                size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=0.1,
                bias_init=Initialization.Zero,
            )
            branches.append(branch_output_layer)
        return nn.LayerList(branches)

    def _mask_branch(
        self, logits: paddle.Tensor, allow_mask: paddle.Tensor
    ) -> paddle.Tensor:
        # Zero out masked logits, then subtract a large value.
        block_mask = -1.0 * allow_mask + 1.0
        logits = logits * allow_mask - 1e8 * block_mask
        return logits

    def _split_masks(self, masks: paddle.Tensor) -> List[paddle.Tensor]:
        split_masks = []
        for idx, _ in enumerate(self.act_sizes):
            start = int(np.sum(self.act_sizes[:idx]))
            end = int(np.sum(self.act_sizes[: idx + 1]))
            split_masks.append(masks[:, start:end])
        return split_masks

    def forward(self, inputs: paddle.Tensor, masks: paddle.Tensor) -> List[DistInstance]:
        # Todo - Support multiple branches in mask code
        branch_distributions = []
        masks = self._split_masks(masks)
        for idx, branch in enumerate(self.branches):
            logits = branch(inputs)
            norm_logits = self._mask_branch(logits, masks[idx])
            distribution = CategoricalDistInstance(norm_logits)
            branch_distributions.append(distribution)
        return branch_distributions
