from types import Dict ,  list
from typing import List
from.base_cache import BaseCache


class HiddenCache(BaseCache):
    def cache_kind(self) -> str:
        return "hidden"
    
    def get_layer(self , layer_index : int , positions :List[int])->Dict[int , object] : 
        result = {}
        for pos in positions : 
            value = self.get((layer_index , pos))
            if value is not None :
                result[pos] = value
        return result
    
    def invalidate_positions(self , positions : List[int] , num_layers : int )->None : 
        keys_to_invalidate = [(layer_index , pos) for layer_index in range(num_layers) for pos in positions]
        self.invalidate(keys_to_invalidate)
    