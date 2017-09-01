#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <sys/time.h>


__host__ void cpu_sort(int * const data, const int num_elements){
  static int cpu_tmp_0[num_elements];
  static int cpu_tmp_1[num_elements];

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

int main(){
  int data[] = {122, 10, 2, 1, 2, 22, 12, 9, 45, 88, 108, 96, 38, 67, 0, 6, 27, 78, 48, 149, 914, 54, 5, 14};
  for (unsigned int i = 0; i< 24; i++){
    printf("%d\t", data[i]);
  }
  printf("\nSerial time:\n");
  struct timeval st; gettimeofday( &st, NULL );
  cpu_sort(data, 24);
  struct timeval et; gettimeofday( &et, NULL );
  printf("%ld ms\n", (et.tv_sec - st.tv_sec) * 1000 + (et.tv_usec - st.tv_usec)/1000);
  for (unsigned int i = 0; i< 24; i++){
    printf("%d\t", data[i]);
  }
  printf("\n");

  return 0;
}
