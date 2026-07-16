#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


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

    // each thread handles 4 FP32 values
    int H4 = H >> 2;


    // one block = one (batch, selected token)
    int token = blockIdx.x;

    int b = token / S;
    int k = token % S;


    // feature index in float4 units
    int h4 =
        blockIdx.y * blockDim.x
        + threadIdx.x;


    if (b >= B || k >= S || h4 >= H4)
        return;



    int src = indices[k];



    const float4* input4 =
        reinterpret_cast<const float4*>(input);


    float4* output4 =
        reinterpret_cast<float4*>(output);



    int input_offset4 =
        b * (N * H4)
        +
        src * H4
        +
        h4;



    int output_offset4 =
        b * (S * H4)
        +
        k * H4
        +
        h4;



    // 16-byte vector load + 16-byte vector store
    output4[output_offset4] =
        input4[input_offset4];

}




void launch_gather_fp32(
    torch::Tensor input,
    torch::Tensor indices,
    torch::Tensor output
)
{

    int B = input.size(0);
    int N = input.size(1);
    int H = input.size(2);
    int S = indices.size(0);



    TORCH_CHECK(
        H % 4 == 0,
        "FP32 float4 gather requires H divisible by 4"
    );



    dim3 block(256);


    dim3 grid(
        B * S,
        (H/4 + 255) / 256
    );



    gather_fp32_kernel<<<grid, block>>>(
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