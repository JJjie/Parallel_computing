//
// Created by 胡文杰 on 2017/8/18.
//

// includes, system
#include <stdio.h>

// includes CUDA Runtime
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions

// 串行处理
__global__ void myhistogram256Kernel_01(
        const unsigned int const * d_hist_data,
        unsigned int * const d_bin_data){
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

    const unsigned int value = d_hist_data[tid];
    atomicAdd(&(d_bin_data[value]), 1);
}

// 数据拆分4路处理
__global__ void myhistogram256Kernel_02(
        const unsigned int const * d_hist_data,
        unsigned int * const d_bin_data){
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

    const unsigned int value_u32 = d_hist_data[tid];

    atomicAdd(&(d_bin_data[((value_u32 & 0x000000FF) )]), 1);
    atomicAdd(&(d_bin_data[((value_u32 & 0x0000FF00) >> 8)]), 1);
    atomicAdd(&(d_bin_data[((value_u32 & 0x00FF0000) >> 16)]), 1);
    atomicAdd(&(d_bin_data[((value_u32 & 0xFF000000) >> 24)]), 1);
}

// 每一个SM计算一个统计直方图，最后汇总
_shared__ unsigned int d_bin_data_shared[256];
__global__ void myhistogram256Kernel_03(
        const unsigned int const * d_hist_data,
        unsigned int * const d_bin_data){
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

    d_bin_data_shared[threadIdx.x] = 0; //初始化共享内存
    const unsigned int value_u32 = d_hist_data[tid];
    __syncthreads();

    atomicAdd(&(d_bin_data[((value_u32 & 0x000000FF) )]), 1);
    atomicAdd(&(d_bin_data[((value_u32 & 0x0000FF00) >> 8)]), 1);
    atomicAdd(&(d_bin_data[((value_u32 & 0x00FF0000) >> 16)]), 1);
    atomicAdd(&(d_bin_data[((value_u32 & 0xFF000000) >> 24)]), 1);

    __syncthreads();
    atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_shared[threadIdx.x]);
}

_shared__ unsigned int d_bin_data_shared[256];
__global__ void myhistogram256Kernel_04(
        const unsigned int const * d_hist_data,
        unsigned int * const d_bin_data, unsigned int N){
    const unsigned int idx = (blockIdx.x * blockDim.x * N) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int tid = idx + idy * blockDim.x * N * gridDim.x;

    d_bin_data_shared[threadIdx.x] = 0; //初始化共享内存

    __syncthreads();
    for (unsigned int i = 0, tid_offset=0; i < N; ++i, tid_offset+=256) {
        const unsigned int value_u32 = d_bin_data[tid+tid_offset];
        atomicAdd(&(d_bin_data[((value_u32 & 0x000000FF) )]), 1);
        atomicAdd(&(d_bin_data[((value_u32 & 0x0000FF00) >> 8)]), 1);
        atomicAdd(&(d_bin_data[((value_u32 & 0x00FF0000) >> 16)]), 1);
        atomicAdd(&(d_bin_data[((value_u32 & 0xFF000000) >> 24)]), 1);
    }
    __syncthreads();

    atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_shared[threadIdx.x]);
}



int main(int argc, char *argv[])
{
    unsigned int *Bin = new unsigned int[256];
    unsigned int Array[] = {1,2,2,3,3,3,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8};
    int *a = 0;
    int *b = 0;
    myhistogram256Kernel_01<<<Array, Bin>>>(a, b);
}
