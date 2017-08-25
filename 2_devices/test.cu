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
  cudaStatus = addWithCuda(c, a, b, arraySize);

  print("%s\n", prop.name);
  print("%d\n", prop.regsPerBlock);
  print("%d\n", prop.wrapSize);
  print("%d\n", prop.clockRate)

  return 0;
}
