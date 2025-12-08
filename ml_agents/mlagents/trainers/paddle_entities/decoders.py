from typing import List, Dict
import paddle
import paddle.nn as nn

# 确保引用的是你之前转换过的 paddle 版本文件
from mlagents.trainers.paddle_entities.layers import linear_layer


class ValueHeads(nn.Layer):
    def __init__(self, stream_names: List[str], input_size: int, output_size: int = 1):
        super().__init__()
        self.stream_names = stream_names
        _value_heads = {}

        for name in stream_names:
            value = linear_layer(input_size, output_size)
            _value_heads[name] = value
        
        # torch.nn.ModuleDict -> paddle.nn.LayerDict
        self.value_heads = nn.LayerDict(_value_heads)

    def forward(self, hidden: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        value_outputs = {}
        for stream_name, head in self.value_heads.items():
            value_outputs[stream_name] = head(hidden).squeeze(-1)
        return value_outputs