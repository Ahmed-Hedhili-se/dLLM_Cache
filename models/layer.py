import torch
import torch.nn as nn
from models.utils import H
from models.rmsnorm import RMSNorm
from models.attention import Attention
from models.moe import MoEBlock

class Layer(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(H)
        self.self_attn = Attention()
        self.post_attention_layernorm = RMSNorm(H)
        self.mlp = MoEBlock()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x