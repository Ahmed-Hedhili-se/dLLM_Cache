#include <torch/extension.h>


torch::Tensor gather_cuda(
    torch::Tensor input,
    torch::Tensor indices
);


PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m
)
{
    m.def(
        "gather",
        &gather_cuda,
        "Half2 CUDA gather"
    );
}