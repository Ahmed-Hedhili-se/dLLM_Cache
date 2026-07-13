import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import H, NE, TOPK
from models.expert import ExpertMLP

class MoEBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, NE, bias=False)
        self.experts = nn.ModuleList([ExpertMLP() for _ in range(NE)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x_flat = x.view(B * T, H)
        routing_weights = F.softmax(self.gate(x_flat), dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, TOPK, dim=-1)
        routing_weights = routing_weights.to(x.dtype)
        out = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=NE).permute(2, 1, 0)
        for expert_idx in range(NE):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            h = self.experts[expert_idx](x_flat[top_x]) * routing_weights[top_x, idx, None]
            out.index_add_(0, top_x, h.to(x.dtype))
        return out.view(B, T, H)