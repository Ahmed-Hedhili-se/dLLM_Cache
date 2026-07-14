"""  Thin wrapper around torch.profiler, used to answer "where is the GPU
(or CPU) actually spending time?" — attention, embeddings, softmax,
MoE routing, etc. Intended to guide later optimization / caching work,
not to run on every benchmark call, since profiling itself has overhead.
"""

from contextlib import contextmanager
from typing import Optional

try:
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class DiffusionProfiler:
    """
    Usage:
        profiler = DiffusionProfiler()
        with profiler.profile():
            engine.generate(prompt)
        profiler.print_summary()
        profiler.export_chrome_trace("trace.json")

    Inside model code, wrap sections with DiffusionProfiler.label("attention")
    (or "moe", "embedding", "softmax", ...) so they show up as separate rows
    in the summary table instead of being lumped into generic ops.
    """

    def __init__(self, use_cuda: Optional[bool] = None):
        if not _HAS_TORCH:
            raise ImportError("profiler.py requires torch")

        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        self.use_cuda = use_cuda

        self.activities = [ProfilerActivity.CPU]
        if self.use_cuda:
            self.activities.append(ProfilerActivity.CUDA)

        self._prof = None

    @contextmanager
    def profile(self, record_shapes: bool = True, profile_memory: bool = True):
        """Context manager: wrap the code you want to profile."""
        self._prof = profile(
            activities=self.activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=False,
        )
        self._prof.__enter__()
        try:
            yield self._prof
        finally:
            self._prof.__exit__(None, None, None)


    @staticmethod
    @contextmanager
    def label(name: str):
        with record_function(name):
            yield


    def print_summary(self, sort_by: str = "cuda_time_total", row_limit: int = 20):
        """Prints a table of the top ops, sorted by time spent."""
        if self._prof is None:
            raise RuntimeError("No profiling data. Call profile() first.")

        sort_key = sort_by if self.use_cuda else "cpu_time_total"
        print(self._prof.key_averages().table(sort_by=sort_key, row_limit=row_limit))


    def export_chrome_trace(self, path: str):
        if self._prof is None:
            raise RuntimeError("No profiling data. Call profile() first.")
        self._prof.export_chrome_trace(path)