import torch
import torch.nn as nn
from models.utils import VS, H, NL, HD, THETA, NH, NE, TOPK, EI, MASK_ID
from models.rmsnorm import RMSNorm
from models.layer import Layer
from models.embedding import TokenEmbedding
from models.rope import build_rope_freqs

class LLaDAMoESmall(nn.Module):

    def __init__(self):
        super().__init__()
        self.embed_tokens = TokenEmbedding()
        self.layers = nn.ModuleList([Layer() for _ in range(NL)])
        self.norm = RMSNorm(H)
        self.lm_head = nn.Linear(H, VS, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = build_rope_freqs(T, HD, THETA, input_ids.device)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.norm(x))
if __name__ == '__main__':
    import sys
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    model = LLaDAMoESmall().to(torch.bfloat16).to(device)
    total = sum((p.numel() for p in model.parameters()))
    print(f'Parameters: {total / 1000000.0:.1f}M')
    print(f'  H={H}, NH={NH}, HD={HD}, NL={NL}, NE={NE}, TOPK={TOPK}, EI={EI}')
    ids = torch.full((1, 32), MASK_ID, dtype=torch.long, device=device)
    ids[0, :16] = torch.randint(0, 1000, (16,))
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (1, 32, VS), f'Unexpected shape: {logits.shape}'
    print(f'Forward pass OK — logits shape: {logits.shape}')
    print(f'  Top-1 predicted token at position 16: {logits[0, 16].argmax().item()}')
    print('LLaDA-MoE-Small: all checks passed.')