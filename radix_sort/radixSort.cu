#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <sys/time.h>


__host__ void cpu_sort(int * const data, const int num_elements){
  int cpu_tmp_0[num_elements];
  int cpu_tmp_1[num_elements];

  for(int bit=0; bit<32; bit++){
    int base_cnt_0 = 0;
    int base_cnt_1 = 0;

    for(int i=0; i<num_elements; i++){
      const int d = data[i];
      const int bit_mask = (1 << bit);
      if ((d & bit_mask) > 0){
        cpu_tmp_1[base_cnt_1] = d;
        base_cnt_1 ++;
      }else{
        cpu_tmp_0[base_cnt_0] = d;
        base_cnt_0 ++;
      }
    }

    for (int i=0; i<base_cnt_0; i++){
      data[i] = cpu_tmp_0[i];
    }

    for(int i=0; i<base_cnt_1; i++){
      data[base_cnt_0 + i] = cpu_tmp_1[i];
    }
  }
}

__global__ void radix_sort(int * const data,
                          const int num_lists, const int num_elements, const int tid,
                          int * const sort_tmp_0, int * const sort_tmp_1){

    for(int bit=0; bit<32; bit++){
      int base_cnt_0 = 0;
      int base_cnt_1 = 0;

      for(int i=0; i<num_elements; i+=num_lists){
        const int d = data[i + tid];
        const int bit_mask = (1 << bit);

        if ((d & bit_mask) > 0){
          sort_tmp_1[base_cnt_1 + tid] = d;
          base_cnt_1 += num_lists;
        }else{
          sort_tmp_0[base_cnt_0 + tid] = d;
          base_cnt_0 += num_lists;
        }
      }

      for (int i=0; i<base_cnt_0; i+=num_lists){
        data[i+tid] = sort_tmp_0[i+tid];
      }

      for(int i=0; i<base_cnt_1; i+=num_lists){
        data[base_cnt_0 + i + tid] = sort_tmp_1[i+tid];
      }
    }
}

cudaError_t sort(int * const data,
                const int size){
    int * dev_data = NULL;
    int * sort_tmp_0 = NULL;
    int * sort_tmp_1 = NULL;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&sort_tmp_1, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&sort_tmp_0, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_data, size * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_data, data, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    radix_sort<<<1, 4>>>(dev_data, 6, size, 0, sort_tmp_0, sort_tmp_1);
    cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(data, dev_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
Error:
    cudaFree(dev_data);
    cudaFree(sort_tmp_0);
    cudaFree(sort_tmp_1);
    return cudaStatus;

}

int main(){
  int data[] = {122, 10, 2, 1, 2, 22, 12, 9, 45, 88, 108, 96, 38, 67, 0, 6, 27, 78, 48, 149, 914, 54, 5, 14};
  struct timeval st, et;

  for (unsigned int i = 0; i< 24; i++){
    printf("%d ", data[i]);
  }
  printf("\n");

  // gettimeofday( &st, NULL );
  // cpu_sort(data, 24);
  // gettimeofday( &et, NULL );
  // for (unsigned int i = 0; i< 24; i++){
  //   printf("%d ", data[i]);
  // }
  // printf("\nSerial time: %ld ms\n", (et.tv_sec - st.tv_sec) * 1000 + (et.tv_usec - st.tv_usec)/1000);

  gettimeofday( &st, NULL );
  cudaError_t cudaStatus = sort(data, 24);
  if (cudaStatus != cudaSuccess)
  {
      fprintf(stderr, "addWithCuda failed!");
      return 1;
  }
  cudaStatus = cudaThreadExit();
  if (cudaStatus != cudaSuccess)
  {
      fprintf(stderr, "cudaThreadExit failed!");
      return 1;
  }
  gettimeofday( &et, NULL );
  for (unsigned int i = 0; i< 24; i++){
    printf("%d ", data[i]);
  }
  printf("\nParallel time: %ld ms\n", (et.tv_sec - st.tv_sec) * 1000 + (et.tv_usec - st.tv_usec)/1000);


  return 0;
}
