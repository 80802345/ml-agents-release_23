import math
from typing import List
import paddle
import paddle.nn as nn

# Ensure you have converted layers.py to paddle_entities
from mlagents.trainers.paddle_entities.layers import (
    linear_layer,
    Swish,
    Initialization,
    LayerNorm,
)


class HyperNetwork(nn.Layer):
    def __init__(
        self, input_size, output_size, hyper_input_size, layer_size, num_layers
    ):
        """
        Hyper Network module. This module will use the hyper_input tensor to generate
        the weights of the main network.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        layer_in_size = hyper_input_size
        layers = []
        for _ in range(num_layers):
            layers.append(
                linear_layer(
                    layer_in_size,
                    layer_size,
                    kernel_init=Initialization.KaimingHeNormal,
                    kernel_gain=1.0,
                    bias_init=Initialization.Zero,
                )
            )
            layers.append(Swish())
            layer_in_size = layer_size
        
        flat_output = linear_layer(
            layer_size,
            input_size * output_size,
            kernel_init=Initialization.KaimingHeNormal,
            kernel_gain=0.1,
            bias_init=Initialization.Zero,
        )

        # Re-initializing the weights of the last layer of the hypernetwork
        # PyTorch: flat_output.weight.data.uniform_(-bound, bound)
        bound = math.sqrt(1 / (layer_size * self.input_size))
        uniform_init = nn.initializer.Uniform(-bound, bound)
        uniform_init(flat_output.weight)

        self.hypernet = nn.Sequential(*layers, LayerNorm(), flat_output)

        # The hypernetwork will not generate the bias of the main network layer
        # PyTorch: self.bias = torch.nn.Parameter(torch.zeros(output_size))
        self.bias = self.create_parameter(
            shape=[output_size],
            default_initializer=nn.initializer.Constant(0.0),
            is_bias=True
        )

    def forward(self, input_activation, hyper_input):
        output_weights = self.hypernet(hyper_input)

        # view -> reshape
        output_weights = output_weights.reshape([-1, self.input_size, self.output_size])

        # torch.bmm -> paddle.bmm
        result = (
            paddle.bmm(input_activation.unsqueeze(1), output_weights).squeeze(1)
            + self.bias
        )
        return result


class ConditionalEncoder(nn.Layer):
    def __init__(
        self,
        input_size: int,
        goal_size: int,
        hidden_size: int,
        num_layers: int,
        num_conditional_layers: int,
        kernel_init: Initialization = Initialization.KaimingHeNormal,
        kernel_gain: float = 1.0,
    ):
        """
        ConditionalEncoder module.
        """
        super().__init__()
        layers: List[nn.Layer] = []
        prev_size = input_size
        for i in range(num_layers):
            if num_layers - i <= num_conditional_layers:
                # This means layer i is a conditional layer since the conditional
                # layers are the last num_conditional_layers
                layers.append(
                    HyperNetwork(prev_size, hidden_size, goal_size, hidden_size, 2)
                )
            else:
                layers.append(
                    linear_layer(
                        prev_size,
                        hidden_size,
                        kernel_init=kernel_init,
                        kernel_gain=kernel_gain,
                    )
                )
            layers.append(Swish())
            prev_size = hidden_size
        # torch.nn.ModuleList -> paddle.nn.LayerList
        self.layers = nn.LayerList(layers)

    def forward(
        self, input_tensor: paddle.Tensor, goal_tensor: paddle.Tensor
    ) -> paddle.Tensor:
        activation = input_tensor
        for layer in self.layers:
            if isinstance(layer, HyperNetwork):
                activation = layer(activation, goal_tensor)
            else:
                activation = layer(activation)
        return activation