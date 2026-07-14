from typing import Callable , List
from .base_cache import BaseCache

class EmbeddingCache(BaseCache):
    def __init__(self , embed_fn : Callable ): 
        super().__init__()
        self.embed_fn = embed_fn


    def cache_kind(self )-> str : 
        return "embedding"
    
    def get_or_compute(self , token_id : int ): 
        cached  = self.get(token_id )
        if cached is not None : 
            return cached
        embedding = self.embed_fn(token_id)
        self.put(token_id, embedding)
        return embedding
    


    def get_or__compute_batch( self , token_ids: List[int] )->List:
        return [self.get_or_compute(token_id) for token_id in token_ids] 
        