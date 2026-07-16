#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>


__global__ void scatter_fp32_kernel(
    float* __restrict__ output,          // [B, N, H], pre-cloned from full_tensor
    const float* __restrict__ partial,   // [B, S, H]
    const int64_t* __restrict__ indices, // [B, S]
    int B,
    int N,
    int H,
    int S
)
{
    int H4 = H >> 2;
    int token = blockIdx.x;   
    int b = token / S;
    int k = token % S;
    int h4 = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= B || k >= S || h4 >= H4) return;

    int64_t dst = indices[(int64_t)b * S + k];
    const float4* partial4 = reinterpret_cast<const float4*>(partial);
    float4* output4 = reinterpret_cast<float4*>(output) ;

    int64_t partial_offset4 = (int64_t)b * S * H4 + (int64_t)k * H4 + h4;

    int64_t output_offset4 = (int64_t)b * N * H4  + dst * H4 + h4;

    output4[output_offset4] = partial4[partial_offset4];
}


void launch_scatter_fp32(
    torch::Tensor output,
    torch::Tensor partial,
    torch::Tensor indices
)
{
    int B = output.size(0);
    int N = output.size(1);
    int H = output.size(2);
    int S = indices.size(1);   

    TORCH_CHECK(
        H % 4 == 0 ,  "FP32 float4 scatter requires H divisible by 4"
    );
    if (S == 0) {
        return;
    }

    dim3 block(256);
    dim3 grid(
        (unsigned int)(B*S) ,  (unsigned int)((H/4+255)/ 256)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    scatter_fp32_kernel<<<grid, block, 0, stream>>>(
        output.data_ptr<float>(),
        partial.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
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
