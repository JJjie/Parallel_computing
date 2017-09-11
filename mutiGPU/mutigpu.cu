#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <sys/time.h>

#define Handle_error(status) (printf("%s\n", cudaGetErrorString(status));return 0;)
#define N 1024 * 1024
#define blockPerGrid 256
#define threadsPerBlock 256

struct DataStruct{
    int deviceID;
    int size;
    int offset;
    float *a;
    float *b;
    float returnValue;
}

void* routine( void *pvoidData){
    DataStruct *data = (DataStruct *)pvoidData;
    if (data->deviceID != 0){
        Handle_error( cudaSetDevice( data->deviceID));
        Handle_error( cudaSetDeviceFlags( cudaDeviceMapHost));
    }

    int size = data->size;
    float *a, *b, v, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = data->a;
    b = data->b;
    partial_c = float[blockPerGrid*sizeof(float)];

    Handle_error( cudaHostGetDevicePointer( &dev_a, a, 0));
    Handle_error( cudaHostGetDevicePointer( &dev_b, b, 0));
    Handle_error( cudaMalloc( (void**)&dev_partial_c, blockPerGrid * sizeof(float)));

    // 计算GPU读取数据的偏移量 a，b
    dev_a += data->offset;
    dev_b += data->offset;

    dot<<<blockPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

    Handle_error( cudaMemcpy(partial_c, dev_partial_c, blockPerGrid*sizeof(float), cudaMemcpyDeviceToHost));

    c = 0;
    for (int i=0; i < blockPerGrid; i++){
        c += partial_c[i];
    }

    Handle_error( cudaFree(dev_partial_c));
    delete [] partial_c;
    data->returnValue = c;
    return 0;
}

int main( void ){
    int deviceCount;
    cudaError_t cudaStatus;

    Handle_error( cudaGetDeviceCount( &deviceCount) );
    if (deviceCount < 2){
        printf("we need at least two compute 1.0 or greater\n");
        return 0;
    }

    cudaDeviceProp prop;
    for(int i=0; i<2; i++){
        Handle_error( cudaGetDeviceProperties( &prop, i) );
        if (prop.canMapHostMemory != 1){
            printf("device %d cannot map memory.\n", i);
            return 0;
        }
    }

    float *a, *b;
    Handle_error( cudaSetDevice(0));
    Handle_error( cudaSetDeviceFlags( cudaDeviceMapHost ));   // 使用零拷贝内存
    Handle_error( cudaHostAlloc( (void**)&a, N*sizeof(float),
                                cudaHostAllocWriteCombined |
                                cudaHostAllocPortable |
                                cudaHostAllocMapped ));
    Handle_error( cudaHostAlloc( (void**)&b, N*sizeof(float),
                                cudaHostAllocWriteCombined |
                                cudaHostAllocPortable |
                                cudaHostAllocMapped ));
    // 用数据填充主机内存
    for( int i=0; i<N; i++){
        a[i] = i;
        b[i] = i * 2;
    }

    // 为使用多线程做好准备
    DataStruct data[2];
    data[0].deviceID = 0;
    data[0].offset = 0;
    data[0].size = N / 2;
    data[0].a = a;
    data[0].b = b;

    data[0].deviceID = 0;
    data[0].offset = N / 2;
    data[0].size = N / 2;
    data[0].a = a;
    data[0].b = b;

    CUTThread thread = start_thread( routine, &(data[1]));
    routine( &(data[0]));
    end_thread( thread );

    Handle_error( cudaFreeHost(a));
    Handle_error( cudaFreeHost(b));

    printf("value calculated: %f\n", data[0].returnValue + data[1].returnValue);
    return 0;

}
