#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>


__global__ void gather_fp32_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ indices,
    float* __restrict__ output,
    int B,
    int N,
    int H,
    int S
)
{
    int H4 = H >> 2;
    int token = blockIdx.x;   // ranges over [0, B*S)
    int b = token / S;
    int k = token % S;
    int h4 = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= B || k >= S || h4 >= H4)
        return;
    int64_t src = indices[(int64_t)b * S + k];
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    int64_t input_offset4 =
        (int64_t)b * N * H4
        + src * H4
        + h4;

    int64_t output_offset4 =
        (int64_t)b * S * H4
        + (int64_t)k * H4
        + h4;

    output4[output_offset4] = input4[input_offset4];
}


void launch_gather_fp32(torch::Tensor input,  torch::Tensor indices, torch::Tensor output)
{
    int B = input.size(0);
    int N = input.size(1);
    int H = input.size(2);
    int S = indices.size(1);   

    TORCH_CHECK(
        H % 4 == 0,
        "FP32 float4 gather requires H divisible by 4"
    );

    // A zero-length selection (S == 0, e.g. V-verify selected nothing this
    // step) is a legal, common case -- launch with 0 blocks and return.
    if (S == 0) {
        return;
    }

    dim3 block(256);
    dim3 grid(
        (unsigned int)(B * S),
        (unsigned int)((H / 4 + 255) / 256)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    gather_fp32_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        B,
        N,
        H,
        S
    );

    cudaError_t err = cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        cudaGetErrorString(err)
    );
}
