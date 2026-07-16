#include <torch/extension.h>
#include "gather_kernel.h"



torch::Tensor gather_forward(
    torch::Tensor input,
    torch::Tensor indices
)
{

    TORCH_CHECK(
        input.is_cuda(),
        "input must be CUDA"
    );


    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32,
        "only FP32 supported"
    );


    auto output = torch::empty(
        {
            input.size(0),
            indices.size(0),
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
        "FP32 optimized CUDA gather"
    );
}