import abc
from typing import Tuple
from enum import Enum
import paddle
import paddle.nn as nn
import math

# 如果你有对应的 paddle 版 model_serialization，请修改这里。
# 否则建议将 is_exporting() 设为默认返回 False 的伪函数。
try:
    from mlagents.trainers.paddle_entities.model_serialization import exporting_to_onnx
except ImportError:
    class exporting_to_onnx:
        @staticmethod
        def is_exporting():
            return False


# class Swish(nn.Layer):
#     def forward(self, data: paddle.Tensor) -> paddle.Tensor:
#         return paddle.multiply(data, paddle.nn.functional.sigmoid(data))

class Swish(nn.Layer):
    def __init__(self):
        super().__init__()
        self._impl = nn.Silu()  # Paddle 内置的 Swish 实现

    def forward(self, data: paddle.Tensor) -> paddle.Tensor:
        return self._impl(data)
    
class Initialization(Enum):
    Zero = 0
    XavierGlorotNormal = 1
    XavierGlorotUniform = 2
    KaimingHeNormal = 3  # also known as Variance scaling
    KaimingHeUniform = 4
    Normal = 5


# Map Enum to Paddle Initializer Classes
def get_initializer(init_type: Initialization):
    if init_type == Initialization.Zero:
        return nn.initializer.Constant(0.0)
    elif init_type == Initialization.XavierGlorotNormal:
        return nn.initializer.XavierNormal()
    elif init_type == Initialization.XavierGlorotUniform:
        return nn.initializer.XavierUniform()
    elif init_type == Initialization.KaimingHeNormal:
        # PyTorch impl uses nonlinearity='linear' in the provided code, which implies gain=1
        # Paddle Kaiming defaults usually assume leaky_relu, but we can stick to standard defaults or adjust
        return nn.initializer.KaimingNormal()
    elif init_type == Initialization.KaimingHeUniform:
        return nn.initializer.KaimingUniform()
    elif init_type == Initialization.Normal:
        return nn.initializer.Normal()
    else:
        return nn.initializer.XavierUniform()


def linear_layer(
    input_size: int,
    output_size: int,
    kernel_init: Initialization = Initialization.XavierGlorotUniform,
    kernel_gain: float = 1.0,
    bias_init: Initialization = Initialization.Zero,
) -> nn.Layer:
    """
    Creates a paddle.nn.Linear layer and initializes its weights.
    """
    # Create layer (Paddle initializes by default, but we will overwrite)
    layer = nn.Linear(input_size, output_size)
    
    # Apply Kernel Init
    initializer = get_initializer(kernel_init)
    initializer(layer.weight)
    
    # Apply Kernel Gain
    if kernel_gain != 1.0:
        with paddle.no_grad():
            layer.weight.scale_(kernel_gain)

    # Apply Bias Init
    bias_initializer = get_initializer(bias_init)
    bias_initializer(layer.bias)
    
    return layer


def lstm_layer(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    batch_first: bool = True,
    forget_bias: float = 1.0,
    kernel_init: Initialization = Initialization.XavierGlorotUniform,
    bias_init: Initialization = Initialization.Zero,
) -> nn.Layer:
    """
    Creates a paddle.nn.LSTM and initializes its weights and biases.
    Paddle nn.LSTM time_major=False is equivalent to batch_first=True.
    """
    lstm = nn.LSTM(
        input_size, 
        hidden_size, 
        num_layers, 
        time_major=not batch_first # Paddle uses time_major
    )

    kernel_initializer = get_initializer(kernel_init)
    bias_initializer = get_initializer(bias_init)

    # Add forget_bias to forget gate bias
    # Paddle LSTM weights are organized: Input, Forget, Cell, Output (Same as PyTorch)
    for name, param in lstm.named_parameters():
        with paddle.no_grad():
            if "weight" in name:
                # Each weight is a concatenation of 4 matrices (I, F, C, O)
                for idx in range(4):
                    block_size = param.shape[0] // 4
                    # Create a temporary view/slice to initialize
                    # Note: Paddle in-place modification on slice can be tricky, 
                    # usually easiest to create the tensor and assign back if possible, 
                    # or use set_value for specific blocks if the initializer supports it.
                    # Here we use a simpler approach: Apply init to a temp tensor and assign.
                    
                    # However, standard initializers operate on the whole Parameter.
                    # Since we need to init sub-blocks, we have to generate data and assign.
                    
                    # Hack: Initialize the whole tensor first if it's the first block, 
                    # but different blocks might need different inits? 
                    # The original code applies the SAME initializer to all blocks.
                    # So we can just initialize the whole tensor at once.
                    pass
                
                # Since the original code applies the SAME kernel_init to all 4 blocks,
                # we can just initialize the entire weight parameter at once.
                kernel_initializer(param)

            if "bias" in name:
                # Initialize all biases
                bias_initializer(param)
                
                # Add forget bias to the Forget Gate (Index 1)
                block_size = param.shape[0] // 4
                start = 1 * block_size
                end = 2 * block_size
                
                # bias_ih and bias_hh in Paddle/PyTorch are separated but structure is same.
                # param[start:end] += forget_bias
                param[start:end] = param[start:end] + forget_bias

    return lstm


class MemoryModule(nn.Layer):
    @abc.abstractproperty
    def memory_size(self) -> int:
        """
        Size of memory that is required at the start of a sequence.
        """
        pass

    @abc.abstractmethod
    def forward(
        self, input_tensor: paddle.Tensor, memories: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Pass a sequence to the memory module.
        :input_tensor: Tensor of shape (batch_size, seq_length, size) that represents the input.
        :memories: Tensor of initial memories.
        :return: Tuple of output, final memories.
        """
        pass


class LayerNorm(nn.Layer):
    """
    A vanilla implementation of layer normalization
    norm_x = (x - mean) / sqrt((x - mean) ^ 2)
    This does not include the trainable parameters gamma and beta for performance speed.
    Typically, this is norm_x * gamma + beta
    """

    def forward(self, layer_activations: paddle.Tensor) -> paddle.Tensor:
        mean = paddle.mean(layer_activations, axis=-1, keepdim=True)
        # Paddle var default unbiased=True, but usually layer norm uses biased variance or simple mean square
        # PyTorch code: mean((x-mean)^2). This is biased variance.
        var = paddle.mean((layer_activations - mean) ** 2, axis=-1, keepdim=True)
        return (layer_activations - mean) / (paddle.sqrt(var + 1e-5))


class LinearEncoder(nn.Layer):
    """
    Linear layers.
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        hidden_size: int,
        kernel_init: Initialization = Initialization.KaimingHeNormal,
        kernel_gain: float = 1.0,
    ):
        super().__init__()
        self.layers = [
            linear_layer(
                input_size,
                hidden_size,
                kernel_init=kernel_init,
                kernel_gain=kernel_gain,
            )
        ]
        self.layers.append(Swish())
        for _ in range(num_layers - 1):
            self.layers.append(
                linear_layer(
                    hidden_size,
                    hidden_size,
                    kernel_init=kernel_init,
                    kernel_gain=kernel_gain,
                )
            )
            self.layers.append(Swish())
        self.seq_layers = nn.Sequential(*self.layers)

    def forward(self, input_tensor: paddle.Tensor) -> paddle.Tensor:
        return self.seq_layers(input_tensor)


class LSTM(MemoryModule):
    """
    Memory module that implements LSTM.
    """

    def __init__(
        self,
        input_size: int,
        memory_size: int,
        num_layers: int = 1,
        forget_bias: float = 1.0,
        kernel_init: Initialization = Initialization.XavierGlorotUniform,
        bias_init: Initialization = Initialization.Zero,
    ):
        super().__init__()
        # We set hidden size to half of memory_size since the initial memory
        # will be divided between the hidden state and initial cell state.
        self.hidden_size = memory_size // 2
        self.lstm = lstm_layer(
            input_size,
            self.hidden_size,
            num_layers,
            batch_first=True, # Paddle equivalent of this param passed to lstm_layer
            forget_bias=forget_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )

    @property
    def memory_size(self) -> int:
        return 2 * self.hidden_size

    def forward(
        self, input_tensor: paddle.Tensor, memories: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:

        if exporting_to_onnx.is_exporting():
            # This transpose is needed both at input and output of the LSTM when
            # exporting because ONNX will expect (sequence_len, batch, memory_size)
            # instead of (batch, sequence_len, memory_size)
            memories = paddle.transpose(memories, perm=[1, 0, 2])

        # h0: (num_layers * num_directions, batch, hidden_size)
        # c0: (num_layers * num_directions, batch, hidden_size)
        # Paddle LSTM requires (h0, c0) as a tuple
        # Assuming memories shape is [1, batch, memory_size] based on single layer assumption in mlagents usually
        
        # We assume memories is [batch, 1, memory_size] or [1, batch, memory_size]? 
        # ML-Agents logic: memories is usually [1, batch, size] for LSTM state if num_layers=1
        
        # Splitting hidden and cell state
        h0 = memories[:, :, : self.hidden_size]
        c0 = memories[:, :, self.hidden_size :]
        
        # Ensure contiguous memory if necessary (clone helps detached form graph or re-layout)
        h0 = h0.clone()
        c0 = c0.clone()

        hidden = (h0, c0)
        lstm_out, (h_out, c_out) = self.lstm(input_tensor, hidden)
        
        # Concatenate hidden and cell state back together
        output_mem = paddle.concat([h_out, c_out], axis=-1)

        if exporting_to_onnx.is_exporting():
            output_mem = paddle.transpose(output_mem, perm=[1, 0, 2])

        return lstm_out, output_mem