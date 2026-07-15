import torch
import unittest
from models.model import LLaDAMoESmall
from cache.embedding_cache import EmbeddingCache
from cache.attention_cache import AttentionCache
from cache.hidden_cache import HiddenCache
from cache.cache_manager import CacheManager
from models.utils import MASK_ID, VS

class TestModelCache(unittest.TestCase):

    def test_caching_flow(self):
        device = 'cpu'
        model = LLaDAMoESmall().to(torch.bfloat16).to(device)
        
        
        k_p = 4
        k_r = 2
        total_steps = 10
        prompt_len = 16
        seq_len = 32
        
        
        cache_manager = CacheManager(k_p=k_p, k_r=k_r, total_steps=total_steps)
        caches = {
            'embed': EmbeddingCache(),
            'layers': [
                {'attn': AttentionCache(), 'mlp': HiddenCache()}
                for _ in range(len(model.layers))
            ]
        }
        input_ids = torch.full((1, seq_len), MASK_ID, dtype=torch.long, device=device)
        input_ids[0, :prompt_len] = torch.randint(0, 1000, (prompt_len,))
        
        # Step 10: Initial step (Full computation & caching)
        print("Running Step 10 (Initial Full Update)...")
        logits_step10 = model(
            input_ids,
            cache_manager=cache_manager,
            caches=caches,
            k_step=10,
            prompt_len=prompt_len,
            update_ratio=0.25
        )
        self.assertEqual(logits_step10.shape, (1, seq_len, VS))
        
        # Step 9: Partial update step
        print("Running Step 9 (Partial Update)...")
        # In a real generator, token values might change
        input_ids[0, prompt_len:] = torch.randint(0, 1000, (seq_len - prompt_len,))
        
        logits_step9 = model(
            input_ids,
            cache_manager=cache_manager,
            caches=caches,
            k_step=9,
            prompt_len=prompt_len,
            update_ratio=0.25
        )
        self.assertEqual(logits_step9.shape, (1, seq_len, VS))
        
        print("Caching flow completed successfully!")

if __name__ == '__main__':
    unittest.main()
