import torch
from typing import Optional, Tuple

from .base_cache import BaseCache
from .selective_compute import scatter_tokens


class AttentionCache:

    def __init__(self):
        self._k = BaseCache()
        self._v = BaseCache()
        self._attn_out = BaseCache()

    @property
    def prompt_len(self) -> int:
        return self._k.prompt_len

    def update_prompt(self, k: torch.Tensor, v: torch.Tensor, attn_out: torch.Tensor, prompt_len: int) -> None:
        self._k.update_prompt(k, prompt_len)
        self._v.update_prompt(v, prompt_len)
        self._attn_out.update_prompt(attn_out, prompt_len)

    def update_response(self, k: torch.Tensor, v: torch.Tensor, attn_out: torch.Tensor) -> None:
        self._k.update_response(k)
        self._v.update_response(v)
        self._attn_out.update_response(attn_out)

    def update_response_partial(
        self, k_r_full: torch.Tensor, new_v: torch.Tensor, partial_attn_out: torch.Tensor, indices: torch.Tensor
    ) -> None:
        """
        Partial cache update after adaptive token selection.

        ``k_r_full`` is the **already-scattered** full-response K tensor
        (built in attention.py before SDPA so it is already correct — storing
        it here avoids a second scatter that was previously wasted).
        V_r is fully replaced (Section A.5).  Only the selected attention
        outputs are scattered into the cached attention-output tensor.
        """
        if self._attn_out.response_cache is None:
            raise RuntimeError("Cannot perform partial update before a full refresh has initialized the cache.")

        # K: store the already-scattered tensor directly (no second scatter)
        self._k.response_cache = k_r_full.detach()
        # V: full replacement
        self._v.update_response(new_v)
        # AttnOut: scatter only the updated positions
        self._attn_out.response_cache = scatter_tokens(
            self._attn_out.response_cache, partial_attn_out, indices
        )




    def get_full(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        k = self._k.get_full()
        v = self._v.get_full()
        attn_out = self._attn_out.get_full()
        if k is None or v is None or attn_out is None:
            return None, None, None
        return k, v, attn_out


    def get_prompt(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        k_p = self._k.get_prompt()
        v_p = self._v.get_prompt()
        attn_out_p = self._attn_out.get_prompt()
        if k_p is None or v_p is None or attn_out_p is None:
            return None, None, None
        return k_p, v_p, attn_out_p

    def get_response(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        k_r = self._k.get_response()
        v_r = self._v.get_response()
        attn_out_r = self._attn_out.get_response()
        if k_r is None or v_r is None or attn_out_r is None:
            return None, None, None
        return k_r, v_r, attn_out_r

    def get_cached_v_response(self) -> Optional[torch.Tensor]:
        return self._v.get_response()

    def reset(self) -> None:
        self._k.reset()
        self._v.reset()
        self._attn_out.reset()