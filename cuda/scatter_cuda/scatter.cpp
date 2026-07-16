#include <torch/extension.h>
#include "scatter_kernel.h"


torch::Tensor scatter_forward(
    torch::Tensor full_tensor,
    torch::Tensor partial_tensor,
    torch::Tensor indices
)
{
    TORCH_CHECK(
        full_tensor.is_cuda(),
        "full_tensor must be a CUDA tensor"
    );

    TORCH_CHECK(
        partial_tensor.is_cuda(),
        "partial_tensor must be a CUDA tensor"
    );

    TORCH_CHECK(
        indices.is_cuda(),
        "indices must be a CUDA tensor"
    );

    TORCH_CHECK(
        full_tensor.device() == partial_tensor.device()
            && full_tensor.device() == indices.device(),
        "full_tensor, partial_tensor, and indices must all be on the same CUDA device"
    );

    TORCH_CHECK(
        full_tensor.scalar_type() == torch::kFloat32,
        "only FP32 supported by this kernel"
    );

    TORCH_CHECK(
        partial_tensor.scalar_type() == torch::kFloat32,
        "partial_tensor must be FP32, matching full_tensor"
    );

    TORCH_CHECK(
        indices.scalar_type() == torch::kInt64,
        "indices must be int64 (torch.long), matching TokenTracker.verify() output"
    );

    TORCH_CHECK(
        full_tensor.dim() == 3,
        "full_tensor must be [B, N, H]"
    );

    TORCH_CHECK(
        partial_tensor.dim() == 3,
        "partial_tensor must be [B, S, H]"
    );

    TORCH_CHECK(
        indices.dim() == 2,
        "indices must be [B, S] -- per-batch destination indices, "
        "matching TokenTracker.verify() / selective_compute.scatter_tokens"
    );

    TORCH_CHECK(
        full_tensor.size(0) == partial_tensor.size(0)
            && full_tensor.size(0) == indices.size(0),
        "full_tensor, partial_tensor, and indices must agree on batch size B"
    );

    TORCH_CHECK(
        partial_tensor.size(1) == indices.size(1),
        "partial_tensor's token dim (S=", partial_tensor.size(1),
        ") must match indices' S dim (", indices.size(1), ")"
    );

    TORCH_CHECK(
        full_tensor.size(2) == partial_tensor.size(2),
        "full_tensor and partial_tensor must have the same feature dim H"
    );

    TORCH_CHECK(
        indices.size(1) <= full_tensor.size(1),
        "S (number of scattered tokens) cannot exceed N (full sequence length)"
    );

    TORCH_CHECK(
        full_tensor.is_contiguous(),
        "full_tensor must be contiguous"
    );

    TORCH_CHECK(
        partial_tensor.is_contiguous(),
        "partial_tensor must be contiguous"
    );

    TORCH_CHECK(
        indices.is_contiguous(),
        "indices must be contiguous"
    );
    auto output = full_tensor.clone();

    launch_scatter_fp32(output, partial_tensor, indices);

    return output;
}


PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m
)
{
    m.def(
        "forward",
        &scatter_forward,
        "FP32 optimized CUDA scatter, per-batch indices [B, S] -> "
        "new [B, N, H] tensor (full_tensor is not modified in place)"
    );
}