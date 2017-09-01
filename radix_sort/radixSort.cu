#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <sys/time>

// template<typename u32 = unsigned int>
__host__ void cpu_sort(u32 * const data, const u32 num_elements){
  static u32 cpu_tmp_0[NUM_ELEM];
  static u32 cpu_tmp_1[NUM_ELEM];

  for(u32 bit=0; bit<32; bit++){
    u32 base_cnt_0 = 0;
    u32 base_cnt_1 = 0;

    for(u32 i=0; i<num_elements; i++){
      const u32 d = data[i];
      const u32 bit_mask = (1 << bit);
      if ((d & bit_mask) > 0){
        cpu_tmp_1[base_cnt_1] = d;
        base_cnt_1 ++;
      }else{
        cpu_tmp_0[base_cnt_0] = d;
        base_cnt_0 ++;
      }
    }

    for (u32 i=0; i<base_cnt_0; i++){
      data[i] = cpu_tmp_0[i];
    }

    for(u32 i=0; i<base_cnt_1; i++){
      data[base_cnt_0 + i] = cpu_tmp_1[i];
    }
  }
}

int main(){
  u32 data[] = {122, 10, 2, 1, 2, 22, 12, 9, 45, 88, 108, 96, 38, 67, 0, 6, 27, 78, 48, 149, 914, 54, 5, 14};
  for (unsigned u32 i = 0; i< 24; i++){
    printf("%d\t", data[i]);
  }
  printf("\nSerial time:\n");
  struct timeval st; gettimeofday( &st, NULL );
  cpu_sort(data, 24);
  struct timeval et; gettimeofday( &et, NULL );
  printf("%ld ms\n", (et.tv_sec - st.tv_sec) * 1000 + (et.tv_usec - st.tv_usec)/1000);
  for (unsigned u32 i = 0; i< 24; i++){
    printf("%d\t", data[i]);
  }
  printf("\n");

  return 0;
}
