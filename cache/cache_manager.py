class CacheManager:

    def __init__(self, k_p: int = 50, k_r: int = 10, total_steps: int = 256):
        self.k_p = k_p
        self.k_r = k_r
        self.total_steps = total_steps


    def is_initial_step(self, k: int) ->bool:
        return k == self.total_steps

    def is_terminal_step(self, k: int) ->bool:
        return k == 0

    def is_prompt_update(self, k: int)-> bool:
        return self.is_initial_step(k )or (k % self.k_p == 0)

    def is_response_full_update(self, k: int) -> bool:
        return self.is_initial_step(k) or (k % self.k_r == 0)

    def is_response_partial_update(self, k: int) -> bool:
        return not self.is_response_full_update(k)