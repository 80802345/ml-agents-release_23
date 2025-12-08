from typing import Tuple, Optional, List

import paddle
import paddle.nn as nn
from mlagents.trainers.exception import UnityTrainerException

# Ensure you have these modules converted as per previous steps
from mlagents.trainers.paddle_entities.layers import (
    LinearEncoder,
    Initialization,
    linear_layer,
    LayerNorm,
)
# Assuming you have a dummy or real implementation of exporting_to_onnx
try:
    from mlagents.trainers.paddle_entities.model_serialization import exporting_to_onnx
except ImportError:
    class exporting_to_onnx:
        @staticmethod
        def is_exporting():
            return False


def get_zero_entities_mask(entities: List[paddle.Tensor]) -> List[paddle.Tensor]:
    """
    Takes a List of Tensors and returns a List of mask Tensor with 1 if the input was
    all zeros (on dimension 2) and 0 otherwise. This is used in the Attention
    layer to mask the padding observations.
    """
    with paddle.no_grad():
        # Generate the masking tensors for each entities tensor (mask only if all zeros)
        # torch.sum(ent**2, axis=2) -> paddle.sum
        key_masks: List[paddle.Tensor] = [
            (paddle.sum(ent**2, axis=2) < 0.01).cast('float32') for ent in entities
        ]
    return key_masks


class MultiHeadAttention(nn.Layer):
    NEG_INF = -1e6

    def __init__(self, embedding_size: int, num_heads: int):
        """
        Multi Head Attention module.
        """
        super().__init__()
        self.n_heads = num_heads
        self.head_size: int = embedding_size // self.n_heads
        self.embedding_size: int = self.head_size * self.n_heads

    def forward(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        n_q: int,
        n_k: int,
        key_mask: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        b = -1  # the batch size

        query = query.reshape(
            [b, n_q, self.n_heads, self.head_size]
        )  # (b, n_q, h, emb / h)
        key = key.reshape([b, n_k, self.n_heads, self.head_size])  # (b, n_k, h, emb / h)
        value = value.reshape(
            [b, n_k, self.n_heads, self.head_size]
        )  # (b, n_k, h, emb / h)

        # permute -> transpose
        query = query.transpose([0, 2, 1, 3])  # (b, h, n_q, emb / h)
        
        # The next few lines are equivalent to : key.permute([0, 2, 3, 1])
        # This is a hack from original ML-Agents to avoid ONNX/Sentis compression issues.
        # We preserve it for compatibility logic.
        key = key.transpose([0, 2, 1, 3])  # (b, h, emb / h, n_k)
        key -= 1
        key += 1
        key = key.transpose([0, 1, 3, 2])  # (b, h, emb / h, n_k)

        qk = paddle.matmul(query, key)  # (b, h, n_q, n_k)

        if key_mask is None:
            qk = qk / (self.embedding_size**0.5)
        else:
            key_mask = key_mask.reshape([b, 1, 1, n_k])
            qk = (1 - key_mask) * qk / (
                self.embedding_size**0.5
            ) + key_mask * self.NEG_INF

        att = paddle.nn.functional.softmax(qk, axis=3)  # (b, h, n_q, n_k)

        value = value.transpose([0, 2, 1, 3])  # (b, h, n_k, emb / h)
        value_attention = paddle.matmul(att, value)  # (b, h, n_q, emb / h)

        value_attention = value_attention.transpose([0, 2, 1, 3])  # (b, n_q, h, emb / h)
        value_attention = value_attention.reshape(
            [b, n_q, self.embedding_size]
        )  # (b, n_q, emb)

        return value_attention, att


class EntityEmbedding(nn.Layer):
    """
    A module used to embed entities before passing them to a self-attention block.
    """

    def __init__(
        self,
        entity_size: int,
        entity_num_max_elements: Optional[int],
        embedding_size: int,
    ):
        """
        Constructs an EntityEmbedding module.
        """
        super().__init__()
        self.self_size: int = 0
        self.entity_size: int = entity_size
        self.entity_num_max_elements: int = -1
        if entity_num_max_elements is not None:
            self.entity_num_max_elements = entity_num_max_elements
        self.embedding_size = embedding_size
        # Initialization scheme
        self.self_ent_encoder = LinearEncoder(
            self.entity_size,
            1,
            self.embedding_size,
            kernel_init=Initialization.Normal,
            kernel_gain=(0.125 / self.embedding_size) ** 0.5,
        )

    def add_self_embedding(self, size: int) -> None:
        self.self_size = size
        self.self_ent_encoder = LinearEncoder(
            self.self_size + self.entity_size,
            1,
            self.embedding_size,
            kernel_init=Initialization.Normal,
            kernel_gain=(0.125 / self.embedding_size) ** 0.5,
        )

    def forward(self, x_self: paddle.Tensor, entities: paddle.Tensor) -> paddle.Tensor:
        num_entities = self.entity_num_max_elements
        if num_entities < 0:
            if exporting_to_onnx.is_exporting():
                raise UnityTrainerException(
                    "Trying to export an attention mechanism that doesn't have a set max \
                    number of elements."
                )
            num_entities = entities.shape[1]

        if self.self_size > 0:
            expanded_self = x_self.reshape([-1, 1, self.self_size])
            # torch.cat([x]*N) equivalent list multiplication
            expanded_self = paddle.concat([expanded_self] * num_entities, axis=1)
            # Concatenate all observations with self
            entities = paddle.concat([expanded_self, entities], axis=2)
        # Encode entities
        encoded_entities = self.self_ent_encoder(entities)
        return encoded_entities


class ResidualSelfAttention(nn.Layer):
    """
    Residual self attention inspired from https://arxiv.org/pdf/1909.07528.pdf.
    """

    EPSILON = 1e-7

    def __init__(
        self,
        embedding_size: int,
        entity_num_max_elements: Optional[int] = None,
        num_heads: int = 4,
    ):
        """
        Constructs a ResidualSelfAttention module.
        """
        super().__init__()
        self.max_num_ent: Optional[int] = None
        if entity_num_max_elements is not None:
            self.max_num_ent = entity_num_max_elements

        self.attention = MultiHeadAttention(
            num_heads=num_heads, embedding_size=embedding_size
        )

        # Initialization scheme
        self.fc_q = linear_layer(
            embedding_size,
            embedding_size,
            kernel_init=Initialization.Normal,
            kernel_gain=(0.125 / embedding_size) ** 0.5,
        )
        self.fc_k = linear_layer(
            embedding_size,
            embedding_size,
            kernel_init=Initialization.Normal,
            kernel_gain=(0.125 / embedding_size) ** 0.5,
        )
        self.fc_v = linear_layer(
            embedding_size,
            embedding_size,
            kernel_init=Initialization.Normal,
            kernel_gain=(0.125 / embedding_size) ** 0.5,
        )
        self.fc_out = linear_layer(
            embedding_size,
            embedding_size,
            kernel_init=Initialization.Normal,
            kernel_gain=(0.125 / embedding_size) ** 0.5,
        )
        self.embedding_norm = LayerNorm()
        self.residual_norm = LayerNorm()

    def forward(self, inp: paddle.Tensor, key_masks: List[paddle.Tensor]) -> paddle.Tensor:
        # Gather the maximum number of entities information
        mask = paddle.concat(key_masks, axis=1)

        inp = self.embedding_norm(inp)
        # Feed to self attention
        query = self.fc_q(inp)  # (b, n_q, emb)
        key = self.fc_k(inp)  # (b, n_k, emb)
        value = self.fc_v(inp)  # (b, n_k, emb)

        # Only use max num if provided
        if self.max_num_ent is not None:
            num_ent = self.max_num_ent
        else:
            num_ent = inp.shape[1]
            if exporting_to_onnx.is_exporting():
                raise UnityTrainerException(
                    "Trying to export an attention mechanism that doesn't have a set max \
                    number of elements."
                )

        output, _ = self.attention(query, key, value, num_ent, num_ent, mask)
        # Residual
        output = self.fc_out(output) + inp
        output = self.residual_norm(output)
        
        # Average Pooling
        # paddle.sum with dim -> axis
        numerator = paddle.sum(output * (1 - mask).reshape([-1, num_ent, 1]), axis=1)
        denominator = paddle.sum(1 - mask, axis=1, keepdim=True) + self.EPSILON
        output = numerator / denominator
        return output