import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import H, EI

class ExpertMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, EI, bias=False)
        self.up_proj = nn.Linear(H, EI, bias=False)
        self.down_proj = nn.Linear(EI, H, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))