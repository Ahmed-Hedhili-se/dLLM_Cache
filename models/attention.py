import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import H, NH, KVH, HD
from models.rmsnorm import RMSNorm
from models.rope import apply_rope

class Attention(nn.Module):

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, NH * HD, bias=False)
        self.k_proj = nn.Linear(H, KVH * HD, bias=False)
        self.v_proj = nn.Linear(H, KVH * HD, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)
        self.q_norm = RMSNorm(HD)
        self.k_norm = RMSNorm(HD)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, NH, HD)
        k = self.k_proj(x).view(B, T, KVH, HD)
        v = self.v_proj(x).view(B, T, KVH, HD)
        q = self.q_norm(q.reshape(-1, HD)).reshape(B, T, NH, HD)
        k = self.k_norm(k.reshape(-1, HD)).reshape(B, T, KVH, HD)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H))