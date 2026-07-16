import torch
import random
import types

import configs.cache_config as cache_config
import configs.inference_config as inference_config
from models.model import LLaDAMoESmall
from models.utils import NL, MASK_ID
from cache import CacheManager, EmbeddingCache, AttentionCache, MoECache
from diffusion.inference import DiffusionInference


def main():
    torch.manual_seed(inference_config.SEED)
    random.seed(inference_config.SEED)

    device = inference_config.DEVICE
    print(f"Device: {device}")
    
    print("Initializing model LLaDAMoESmall...")
    model = LLaDAMoESmall()
    if inference_config.USE_BFLOAT16:
        model = model.to(torch.bfloat16)
    model = model.to(device)
    model.eval()

    cache_manager = CacheManager(
        k_p=cache_config.KP,
        k_r=cache_config.KR,
        total_steps=cache_config.NUM_DIFFUSION_STEPS
    )

    caches = {'embed': EmbeddingCache() if cache_config.ENABLE_EMBEDDING_CACHE else None}
    layer_caches = []
    for _ in range(NL):
        layer_caches.append({
            'attn': AttentionCache() if cache_config.ENABLE_ATTENTION_CACHE else None,
            'mlp': MoECache() if cache_config.ENABLE_MOE_CACHE else None
        })
    caches['layers'] = layer_caches

    prompt_ids = torch.randint(0, 1000, (16,)).tolist()
    prompt_len = len(prompt_ids)

    inference = DiffusionInference(
        model=model,
        tokenizer=None,
        total_steps=cache_config.NUM_DIFFUSION_STEPS,
        mask_token_id=MASK_ID,
        sampling_strategy="greedy"
    )

    # Hook denoise_step to capture the current step
    original_denoise = inference.generator._denoise_step
    def hooked_denoise(self, tokens, step, editable):
        self._current_step = step
        return original_denoise(tokens, step, editable)
    inference.generator._denoise_step = types.MethodType(hooked_denoise, inference.generator)

    # Override _model_fn to pass cache parameters and the k_step state
    def custom_model_fn(self, tokens: list):
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
        k_step = cache_config.NUM_DIFFUSION_STEPS - getattr(self.generator, "_current_step", 0)
        
        with torch.no_grad():
            output = self.model(
                input_ids,
                cache_manager=cache_manager,
                caches=caches,
                k_step=k_step,
                prompt_len=prompt_len,
                update_ratio=cache_config.UPDATE_RATIO
            )
            logits = output.logits if hasattr(output, "logits") else output
        return logits[0]

    inference._model_fn = types.MethodType(custom_model_fn, inference)

    print(f"Generating {inference_config.MAX_NEW_TOKENS} new tokens...")
    output_ids = inference.generate(prompt_ids, gen_length=inference_config.MAX_NEW_TOKENS)
    print("Generation complete.")
    print("Output sequence length:", len(output_ids))

if __name__ == "__main__":
    main()
