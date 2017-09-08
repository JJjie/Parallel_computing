#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <sys/time.h>

#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);


__global__ void dot(float *a, float *b, float *c){
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N){
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    // 对线程块的线程进行同步,防止写后读bug
    __syncthreads();
    // 对于归约运算，以下代码要求threadsPerBlock必须是2的整数倍
    int i = blockDim.x / 2;
    while (i != 0){
        if (cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

int main(){
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    cudaError_t cudastatus;

    a = new float[N];
    b = new float[N];
    partial_c = new float[blocksPerGrid];

    cudastatus = cudaMalloc((void**)&dev_a, N*sizeof(float));
    if (cudastatus != cudaSuccess){
        printf("cuda malloc fail!\n");
        goto Error;
    }
    cudastatus = cudaMalloc((void**)&dev_b, N*sizeof(float));
    if (cudastatus != cudaSuccess){
        printf("cuda malloc fail!\n");
        goto Error;
    }
    cudastatus = cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float));
    if (cudastatus != cudaSuccess){
        printf("cuda malloc fail!\n");
        goto Error;
    }

    for(int i=0; i<N; i++){
        a[i] = i;
        b[i] = i * 2;
    }

    cudastatus = cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    if (cudastatus != cudaSuccess){
        printf("cuda memcpy fail!\n");
        goto Error;
    }
    cudastatus = cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
    if (cudastatus != cudaSuccess){
        printf("cuda memcpy fail!\n");
        goto Error;
    }

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    cudastatus = cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
    if (cudastatus != cudaSuccess){
        printf("cuda memcpy fail!\n");
        goto Error;
    }

    c = 0;
    for (int i=0; i<blocksPerGrid: i++) {
        c += partial_c[i];
    }

    #define sum_squares(x) (x * (x+1) * (2*x+1) / 6)
    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float)(N-1)));

Error:
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    delete [] a;
    delete [] b;
    delete [] partial_c;


    return 0;
}
