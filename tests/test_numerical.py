import torch
from models.model import LLaDAMoESmall
from cache.cache_manager import CacheManager
from cache.embedding_cache import EmbeddingCache
from cache.attention_cache import AttentionCache
from cache.hidden_cache import HiddenCache

def test_numerical_consistency():
    torch.manual_seed(42)
    device = 'cpu'
    model = LLaDAMoESmall().to(torch.float32).to(device)
    model.eval()

    prompt_len = 16
    seq_len = 32

    input_ids = torch.randint(0, 1000, (1, seq_len), device=device)

    # 1. Full Recomputation (No Cache)
    with torch.no_grad():
        out_no_cache = model(input_ids)

    # 2. Setup Cache
    cache_manager = CacheManager(k_p=1, k_r=10, total_steps=10)
    caches = {
        'embed': EmbeddingCache(),
        'layers': [{'attn': AttentionCache(), 'mlp': HiddenCache()} for _ in range(4)]
    }

    # Step 10: Initialize cache
    with torch.no_grad():
        out_init_cache = model(
            input_ids, 
            cache_manager=cache_manager, 
            caches=caches, 
            k_step=10, 
            prompt_len=prompt_len
        )
        
    print(f"Initial cache output matches full compute: {torch.allclose(out_no_cache, out_init_cache)}")

    # Step 9: Adaptive Update with 100% update ratio (should be identical to full compute!)
    input_ids_new = torch.randint(0, 1000, (1, seq_len), device=device)
    input_ids_new[0, :prompt_len] = input_ids[0, :prompt_len] # Prompt doesn't change
    
    with torch.no_grad():
        out_no_cache_new = model(input_ids_new)
        
        out_partial_100 = model(
            input_ids_new, 
            cache_manager=cache_manager, 
            caches=caches, 
            k_step=9, 
            prompt_len=prompt_len,
            update_ratio=1.0  # Force 100% update
        )
        
    print(f"Partial update (100% ratio) matches full compute: {torch.allclose(out_no_cache_new, out_partial_100)}")
    diff = (out_no_cache_new - out_partial_100).abs().max()
    print(f"Max diff: {diff}")

if __name__ == '__main__':
    test_numerical_consistency()
