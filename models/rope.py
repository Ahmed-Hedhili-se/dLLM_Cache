import torch

def build_rope_freqs(max_seq: int, head_dim: int, theta: float, device):
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    pos = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return (emb.cos(), emb.sin())

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = (x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:])
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)