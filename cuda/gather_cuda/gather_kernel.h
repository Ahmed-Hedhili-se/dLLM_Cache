#pragma once
#include <torch/extension.h>
void launch_gather_fp32(
    torch::Tensor input,
    torch::Tensor indices,
    torch::Tensor output
);
