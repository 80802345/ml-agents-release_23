from typing import Tuple, Optional, Union
from math import floor

import paddle
import paddle.nn as nn

# Ensure you have converted layers.py
from mlagents.trainers.paddle_entities.layers import linear_layer, Initialization, Swish


class Normalizer(nn.Layer):
    def __init__(self, vec_obs_size: int):
        super().__init__()
        # register_buffer persists state but doesn't add to gradient computation
        self.register_buffer("normalization_steps", paddle.to_tensor([1], dtype='float32'))
        self.register_buffer("running_mean", paddle.zeros([vec_obs_size], dtype='float32'))
        self.register_buffer("running_variance", paddle.ones([vec_obs_size], dtype='float32'))

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        normalized_state = paddle.clip(
            (inputs - self.running_mean)
            / paddle.sqrt(self.running_variance / self.normalization_steps),
            -5,
            5,
        )
        return normalized_state

    def update(self, vector_input: paddle.Tensor) -> None:
        with paddle.no_grad():
            steps_increment = vector_input.shape[0]
            total_new_steps = self.normalization_steps + steps_increment

            input_to_old_mean = vector_input - self.running_mean
            # axis=0 sum
            new_mean = self.running_mean + (
                input_to_old_mean / total_new_steps
            ).sum(0)

            input_to_new_mean = vector_input - new_mean
            new_variance = self.running_variance + (
                input_to_new_mean * input_to_old_mean
            ).sum(0)

            # Update references.
            # In Paddle, assigning to self.running_mean directly updates the member.
            # To ensure it stays a buffer/parameter type if needed, assign is safer.
            paddle.assign(new_mean, self.running_mean)
            paddle.assign(new_variance, self.running_variance)
            paddle.assign(paddle.to_tensor([total_new_steps], dtype='float32'), self.normalization_steps)

    def copy_from(self, other_normalizer: "Normalizer") -> None:
        paddle.assign(other_normalizer.normalization_steps, self.normalization_steps)
        paddle.assign(other_normalizer.running_mean, self.running_mean)
        paddle.assign(other_normalizer.running_variance, self.running_variance)


def conv_output_shape(
    h_w: Tuple[int, int],
    kernel_size: Union[int, Tuple[int, int]] = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tuple[int, int]:
    """
    Calculates the output shape (height and width) of the output of a convolution layer.
    """
    if not isinstance(kernel_size, tuple):
        kernel_size = (int(kernel_size), int(kernel_size))
    h = floor(
        ((h_w[0] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((h_w[1] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w


def pool_out_shape(h_w: Tuple[int, int], kernel_size: int) -> Tuple[int, int]:
    """
    Calculates the output shape (height and width) of the output of a max pooling layer.
    """
    height = (h_w[0] - kernel_size) // 2 + 1
    width = (h_w[1] - kernel_size) // 2 + 1
    return height, width


class VectorInput(nn.Layer):
    def __init__(self, input_size: int, normalize: bool = False):
        super().__init__()
        self.normalizer: Optional[Normalizer] = None
        if normalize:
            self.normalizer = Normalizer(input_size)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        if self.normalizer is not None:
            inputs = self.normalizer(inputs)
        return inputs

    def copy_normalization(self, other_input: "VectorInput") -> None:
        if self.normalizer is not None and other_input.normalizer is not None:
            self.normalizer.copy_from(other_input.normalizer)

    def update_normalization(self, inputs: paddle.Tensor) -> None:
        if self.normalizer is not None:
            self.normalizer.update(inputs)


class FullyConnectedVisualEncoder(nn.Layer):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        self.output_size = output_size
        self.input_size = height * width * initial_channels
        self.dense = nn.Sequential(
            linear_layer(
                self.input_size,
                self.output_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,  # Use ReLU gain
            ),
            nn.LeakyReLU(),
        )

    def forward(self, visual_obs: paddle.Tensor) -> paddle.Tensor:
        hidden = visual_obs.reshape([-1, self.input_size])
        return self.dense(hidden)


class SmallVisualEncoder(nn.Layer):
    """
    CNN architecture used by King in their Candy Crush predictor
    """

    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        self.h_size = output_size
        conv_1_hw = conv_output_shape((height, width), 3, 1)
        conv_2_hw = conv_output_shape(conv_1_hw, 3, 1)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 144

        self.conv_layers = nn.Sequential(
            nn.Conv2D(initial_channels, 35, kernel_size=[3, 3], stride=[1, 1]),
            nn.LeakyReLU(),
            nn.Conv2D(35, 144, kernel_size=[3, 3], stride=[1, 1]),
            nn.LeakyReLU(),
        )
        self.dense = nn.Sequential(
            linear_layer(
                self.final_flat,
                self.h_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,  # Use ReLU gain
            ),
            nn.LeakyReLU(),
        )

    def forward(self, visual_obs: paddle.Tensor) -> paddle.Tensor:
        hidden = self.conv_layers(visual_obs)
        hidden = hidden.reshape([-1, self.final_flat])
        return self.dense(hidden)


class SimpleVisualEncoder(nn.Layer):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        self.h_size = output_size
        conv_1_hw = conv_output_shape((height, width), 8, 4)
        conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32

        self.conv_layers = nn.Sequential(
            nn.Conv2D(initial_channels, 16, kernel_size=[8, 8], stride=[4, 4]),
            nn.LeakyReLU(),
            nn.Conv2D(16, 32, kernel_size=[4, 4], stride=[2, 2]),
            nn.LeakyReLU(),
        )
        self.dense = nn.Sequential(
            linear_layer(
                self.final_flat,
                self.h_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,  # Use ReLU gain
            ),
            nn.LeakyReLU(),
        )

    def forward(self, visual_obs: paddle.Tensor) -> paddle.Tensor:
        hidden = self.conv_layers(visual_obs)
        hidden = hidden.reshape([-1, self.final_flat])
        return self.dense(hidden)


class NatureVisualEncoder(nn.Layer):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        self.h_size = output_size
        conv_1_hw = conv_output_shape((height, width), 8, 4)
        conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
        conv_3_hw = conv_output_shape(conv_2_hw, 3, 1)
        self.final_flat = conv_3_hw[0] * conv_3_hw[1] * 64

        self.conv_layers = nn.Sequential(
            nn.Conv2D(initial_channels, 32, kernel_size=[8, 8], stride=[4, 4]),
            nn.LeakyReLU(),
            nn.Conv2D(32, 64, kernel_size=[4, 4], stride=[2, 2]),
            nn.LeakyReLU(),
            nn.Conv2D(64, 64, kernel_size=[3, 3], stride=[1, 1]),
            nn.LeakyReLU(),
        )
        self.dense = nn.Sequential(
            linear_layer(
                self.final_flat,
                self.h_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,  # Use ReLU gain
            ),
            nn.LeakyReLU(),
        )

    def forward(self, visual_obs: paddle.Tensor) -> paddle.Tensor:
        hidden = self.conv_layers(visual_obs)
        hidden = hidden.reshape([-1, self.final_flat])
        return self.dense(hidden)


class ResNetBlock(nn.Layer):
    def __init__(self, channel: int):
        """
        Creates a ResNet Block.
        """
        super().__init__()
        self.layers = nn.Sequential(
            Swish(),
            nn.Conv2D(channel, channel, kernel_size=[3, 3], stride=[1, 1], padding=1),
            Swish(),
            nn.Conv2D(channel, channel, kernel_size=[3, 3], stride=[1, 1], padding=1),
        )

    def forward(self, input_tensor: paddle.Tensor) -> paddle.Tensor:
        return input_tensor + self.layers(input_tensor)


class ResNetVisualEncoder(nn.Layer):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        n_channels = [16, 32, 32]  # channel for each stack
        n_blocks = 2  # number of residual blocks
        layers = []
        last_channel = initial_channels
        for _, channel in enumerate(n_channels):
            layers.append(nn.Conv2D(last_channel, channel, kernel_size=[3, 3], stride=[1, 1], padding=1))
            layers.append(nn.MaxPool2D(kernel_size=[3, 3], stride=[2, 2]))
            height, width = pool_out_shape((height, width), 3)
            for _ in range(n_blocks):
                layers.append(ResNetBlock(channel))
            last_channel = channel
        layers.append(Swish())
        self.final_flat_size = n_channels[-1] * height * width
        self.dense = linear_layer(
            self.final_flat_size,
            output_size,
            kernel_init=Initialization.KaimingHeNormal,
            kernel_gain=1.41,  # Use ReLU gain
        )
        self.sequential = nn.Sequential(*layers)

    def forward(self, visual_obs: paddle.Tensor) -> paddle.Tensor:
        hidden = self.sequential(visual_obs)
        before_out = hidden.reshape([-1, self.final_flat_size])
        return paddle.nn.functional.relu(self.dense(before_out))