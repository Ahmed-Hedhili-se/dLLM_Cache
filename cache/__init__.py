"""
Caching infrastructure for masked-diffusion LM inference (e.g.
LLaDA-MoE). Tracks which token positions actually changed between
diffusion steps and lets the model skip recomputation for positions
that didn't, at the embedding, hidden-state, attention, and MoE
levels. CacheManager (cache_manager.py) is the main entry point.
"""

from .token_tracker import TokenTracker
from .base_cache import BaseCache
from .embedding_cache import EmbeddingCache
from .hidden_cache import HiddenCache
from .attention_cache import AttentionCache
from .moe_cache import MoECache
from .selective_compute import gather_tokens, scatter_tokens
from .cache_manager import CacheManager

__all__ = [
    "TokenTracker",
    "BaseCache",
    "EmbeddingCache",
    "HiddenCache",
    "AttentionCache",
    "MoECache",
    "gather_tokens",
    "scatter_tokens",
    "CacheManager",
]