#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 32
#define MAX_HEAD_DIM 128

__global__
void attention_kernel_v10(
    const float* Q,
    const float* K,
    const float* V,
    float* Output,  int T, int HeadDim)
{
    int row  = blockIdx.x;
    int lane = threadIdx.x;

    if (row >= T)  return;

    __shared__ float sharedK[TILE_SIZE][MAX_HEAD_DIM];
    __shared__ float sharedV[TILE_SIZE][MAX_HEAD_DIM];


    float q_reg[MAX_HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HeadDim; d++)
    {
        q_reg[d] = Q[row * HeadDim + d];
    }


    float out_reg[MAX_HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HeadDim; d++)
    {
        out_reg[d] = 0.0f;
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f ;

    for (int tile = 0; tile < T; tile += TILE_SIZE)
    {
        int col = tile + lane;

        int thread_id = threadIdx.x;
        int tile_elements = TILE_SIZE * HeadDim;


        for (int idx = thread_id; idx < tile_elements ; idx += blockDim.x)
        {
            int local_row = idx / HeadDim;
            int local_col = idx % HeadDim;

            int global_row = tile + local_row;

            if (global_row < T)
            {
                sharedK[local_row][local_col] =
                    K[global_row * HeadDim + local_col];

                sharedV[local_row][local_col] =
                    V[global_row * HeadDim + local_col];
            }


        }

        __syncthreads();


        float score = -INFINITY;

        if (col < T)
        {
            score = 0.0f;

#pragma unroll
            for (int d = 0; d < HeadDim; d++)
            {
                score += q_reg[d] * sharedK[lane][d];
            }

        }


        float tile_max = score;

        for (int offset = 16; offset > 0; offset /= 2){
            tile_max = fmaxf(tile_max ,  __shfl_down_sync(0xffffffff, tile_max, offset) );
        }

        tile_max = __shfl_sync(0xffffffff, tile_max, 0);

        float new_max = fmaxf(running_max, tile_max);

        float scale = expf(running_max - new_max);


#pragma unroll

        for (int d = 0; d < HeadDim; d++)   
            out_reg[d] *= scale;

        
        running_sum *= scale;

        float local_exp =
            (col < T)
                ? expf(score - new_max)
                : 0.0f;

        float tile_sum = local_exp;

        for (int offset = 16; offset > 0; offset >>= 1)
            tile_sum += __shfl_down_sync(0xffffffff ,tile_sum, offset);

        tile_sum = __shfl_sync(0xffffffff , tile_sum ,0);

        running_sum += tile_sum;
        running_max = new_max;

        if (col < T)
        {
            float probability =local_exp / running_sum;
#pragma unroll
            for (int d = 0; d < HeadDim; d++)
            {
                out_reg[d] += probability *sharedV[lane][d];
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int d = 0; d < HeadDim; d++)
        Output[row * HeadDim + d] = out_reg[d];

        

}