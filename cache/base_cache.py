from abc import ABC , abstractmethod
from typing import Any , Iterable , Optional

class BaseCache(ABC): 
    def __init__(self):
        self._store: dict= {}
    
    def get (self  , key : Any)-> Optional[Any] : 
        return self._store.get(key)
    
    def set(self, key : Any , value :Any)->None : 
        self._store[key]= value

    def contains(self , key : Any )->bool: 
        return key in self._store
    
    def invalidate(self , keys:Iterable[Any]) -> None : 
        for key in keys : 
            self._store.pop(key  , None )
    
    def clear(self )->None : 
        self._store.clear()


    def __len__(self) ->int : 
        return len(self._store)
    

    @abstractmethod
    def cache_kind(self) -> str:
        raise NotImplementedError