#include "cuda_runtime.h"           //CUDA运行时API
#include "device_launch_parameters.h"
#include <stdio.h>

int main(){

  cudaError_t cudaStatus;
  int num = 0;
  cudaDeviceProp prop;
  cudaStatus = cudaGetDeviceCount(&num);
  for(int i = 0;i<num;i++)
  {
      cudaGetDeviceProperties(&prop,i);
  }
  // cudaStatus = addWithCuda(c, a, b, arraySize);

  printf("%s\n", prop.name);
  printf("%d\n", prop.regsPerBlock);
  printf("%d\n", prop.wrapSize);
  printf("%d\n", prop.clockRate)

  return 0;
}
