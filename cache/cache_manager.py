class CacheManager:
    """
    Manages cache refresh scheduling for dLLM-Cache.

    k_step convention: k_step = total_steps - diffusion_step.
      - k_step == total_steps  → initial step (diffusion_step == 0)
      - k_step == 1            → last step    (diffusion_step == total_steps - 1)

    Refresh intervals (k_p, k_r) are expressed in **diffusion steps**, so we
    convert k_step → diffusion_step before the modulo check.  This guarantees
    that refreshes fire at equally-spaced intervals (every k_p / k_r steps),
    avoiding the asymmetry that arises from doing ``k % k_p`` directly on the
    countdown value.
    """

    def __init__(self, k_p: int = 50, k_r: int = 10, total_steps: int = 256):
        self.k_p = k_p
        self.k_r = k_r
        self.total_steps = total_steps

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _to_step(self, k: int) -> int:
        """Convert k_step (countdown) to 0-indexed diffusion step."""
        return self.total_steps - k

    def is_initial_step(self, k: int) -> bool:
        """True only for the very first diffusion step (step == 0)."""
        return k == self.total_steps

    def is_terminal_step(self, k: int) -> bool:
        """True only for the very last diffusion step."""
        return k == 1

    # ------------------------------------------------------------------
    # refresh predicates
    # ------------------------------------------------------------------

    def is_prompt_update(self, k: int) -> bool:
        """Refresh prompt cache at step 0 and then every k_p diffusion steps."""
        if self.is_initial_step(k):
            return True
        step = self._to_step(k)   # 0-indexed diffusion step
        return (step % self.k_p) == 0

    def is_response_full_update(self, k: int) -> bool:
        """Full response refresh at step 0 and then every k_r diffusion steps."""
        if self.is_initial_step(k):
            return True
        step = self._to_step(k)
        return (step % self.k_r) == 0

    def is_response_partial_update(self, k: int) -> bool:
        """Partial (adaptive) response update on all other steps."""
        return not self.is_response_full_update(k)