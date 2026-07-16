#include <torch/extension.h>
#include "gather_kernel.h"


torch::Tensor gather_forward(
    torch::Tensor input,
    torch::Tensor indices
)
{
    TORCH_CHECK(
        input.is_cuda(),
        "input must be a CUDA tensor"
    );

    TORCH_CHECK(
        indices.is_cuda(),
        "indices must be a CUDA tensor"
    );

    TORCH_CHECK(
        input.device() == indices.device(),
        "input and indices must be on the same CUDA device"
    );

    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32,
        "only FP32 supported by this kernel"
    );

    TORCH_CHECK(
        indices.scalar_type() == torch::kInt64,
        "indices must be int64 (torch.long), matching TokenTracker.verify() output"
    );

    TORCH_CHECK(
        input.dim() == 3,
        "input must be [B, N, H]"
    );

    TORCH_CHECK(
        indices.dim() == 2,
        "indices must be [B, S] -- per-batch selected indices, "
        "matching TokenTracker.verify() / selective_compute.gather_tokens. "
        "A flat [S] tensor shared across the batch is not supported."
    );

    TORCH_CHECK(
        input.size(0) == indices.size(0),
        "input and indices must agree on batch size B (got input B=",
        input.size(0), ", indices B=", indices.size(0), ")"
    );

    TORCH_CHECK(
        input.is_contiguous(),
        "input must be contiguous -- call .contiguous() first "
        "(a .transpose()/slice result, common in attention.py, is not contiguous)"
    );

    TORCH_CHECK(
        indices.is_contiguous(),
        "indices must be contiguous"
    );

    auto output = torch::empty(
        {
            input.size(0),
            indices.size(1),   
            input.size(2)
        },
        input.options()
    );

    launch_gather_fp32(
        input,
        indices,
        output
    );

    return output;
}


PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m
)
{
    m.def(
        "forward",
        &gather_forward,
        "FP32 optimized CUDA gather, per-batch indices [B, S] -> output [B, S, H]"
    );
}