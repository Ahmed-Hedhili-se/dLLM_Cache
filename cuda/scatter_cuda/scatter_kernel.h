#pragma once

#include <torch/extension.h>

void launch_scatter_fp32(
    torch::Tensor output,
    torch::Tensor partial,
    torch::Tensor indices
);