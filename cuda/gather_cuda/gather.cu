#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>



__global__ void gather_half2_kernel(
    const half* __restrict__ input,
    const int64_t* __restrict__ indices,
    half* __restrict__ output,
    int B,
    int N,
    int H,
    int S
)
{
    int H2 = H >> 1;
    int bid = blockIdx.x;
    int b = bid / S;
    int k = bid % S;


    int h2 =
        blockIdx.y * blockDim.x
        + threadIdx.x;



    if (b >= B || k >= S || h2 >= H2) return;
    int src = indices[k];

    const half2* input2 =
        reinterpret_cast<const half2*>(input);


    half2* output2 =
        reinterpret_cast<half2*>(output);


    int input_offset =b * (N * H2)+src * H2+h2;

    int output_offset =b * (S * H2)+k * H2+h2;
    output2[output_offset] =
        input2[input_offset];

}
torch::Tensor gather_cuda(
    torch::Tensor input,
    torch::Tensor indices
)
{

    TORCH_CHECK(
        input.is_cuda(),
        "input must be CUDA"
    );


    TORCH_CHECK(
        indices.is_cuda(),
        "indices must be CUDA"
    );


    TORCH_CHECK(
        input.dtype() == torch::kFloat16,
        "only FP16 supported"
    );


    TORCH_CHECK(
        input.dim() == 3,
        "input must be [B,N,H]"
    );


    TORCH_CHECK(
        indices.dim() == 1,
        "indices must be [S]"
    );


    TORCH_CHECK(
        input.is_contiguous(),
        "input must be contiguous"
    );


    int B = input.size(0);
    int N = input.size(1);
    int H = input.size(2);

    int S = indices.size(0);



    TORCH_CHECK(
        H % 2 == 0,
        "H must be divisible by 2"
    );


    auto output =
        torch::empty(
            {B,S,H},
            input.options()
        );



    int threads = 256;


    dim3 block(threads);



    dim3 grid(
        B*S,
        (H/2 + threads - 1)/threads
    );



    cudaStream_t stream =
        at::cuda::getDefaultCUDAStream();



    gather_half2_kernel<<<
        grid,
        block,
        0,
        stream
    >>>(
        (half*)input.data_ptr<at::Half>(),
        indices.data_ptr<int64_t>(),
        (half*)output.data_ptr<at::Half>(),
        B,
        N,
        H,
        S
    );



    return output;
}