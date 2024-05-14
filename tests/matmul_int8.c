/*
 * Copyright (C) 2024  Jasbir Matharu, <jasjnuk@gmail.com>
 *
 * This file is part of rk3588-npu.
 *
 * rk3588-npu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * rk3588-npu is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with rk3588-npu.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>

#include <libdrm/drm.h>

#include "rknpu-ioctl.h"
#include "npu_interface.h"
#include "npu_matmul.h"

#define MAX_M 544
#define MAX_K 4096 
#define MAX_N 4096 

  // Test currently runs against kernel 5.10 haven't tested 6.1 kernel.

  // matrix A max size
  int8_t matrixA[(MAX_M*MAX_K)];

  // matrix B max size
  int8_t matrixB[(MAX_N*MAX_K)];

  // matrix C max size
  int32_t expected_result[MAX_M*MAX_N];

  uint64_t npu_regs[112];

void matmul_int(int m, int k, int n, int8_t *src0 , int8_t *src1, int32_t* dst) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0;
      for (int l = 0; l < k; l++) {
        sum += (int32_t) (src0[i*k + l] * src1[j*k + l]);
      }
     dst[i*n + j] = sum;
    }
  }
}

int8_t rand_int() {
  return (int8_t)((rand()%255)+1);
}

int main(int argc, char **argv) {

  unsigned int M=0;
  unsigned int K=0;
  unsigned int N=0;

  int ret=0;

  if (argc !=4) {
    printf("Invalid number of args %d, needs to supply M K N ie matmul_fp16 <M> <K> <N>\n",argc);
    return -1;
  }

  M = atoi(argv[1]);
  K = atoi(argv[2]);
  N = atoi(argv[3]);

  if ((M<=0) || (M>MAX_M) | (((M%4)!=0) && (M!=1))) {
    printf("M [%d] is out of range or not a mutliple of 4 \n",M);
    return -1;
  }

  if ((K<=0) || (K>MAX_K) || ((K%32) != 0)) {
    printf("K [%d] is out of range or not a mutliple of 32\n",K);
    return -1;
  }

  if ((N<=0) || (N>MAX_N) || ((N%16) != 0)) {
    printf("N [%d] is out of range or not a mutliple of 32\n",N);
    return -1;
  }

  // Open DRI called "rknpu"
  int fd = npu_open();

  uint64_t regcmd_dma, regcmd_obj;
  uint32_t regcmd_handle;
  uint64_t *regcmd = mem_allocate(fd, 1024, &regcmd_dma, &regcmd_obj, 0, &regcmd_handle);

  uint64_t tasks_dma, tasks_obj;
  uint32_t tasks_handle;
  struct rknpu_task *tasks = mem_allocate(fd, 1024, &tasks_dma, &tasks_obj, RKNPU_MEM_KERNEL_MAPPING, &tasks_handle);

  uint64_t input_dma, input_obj;
  uint32_t input_handle;
  void *input = mem_allocate(fd, M*K*sizeof(int8_t), &input_dma, &input_obj, 0, &input_handle);

  uint64_t weights_dma, weights_obj;
  uint32_t weights_handle;
  void *weights = mem_allocate(fd, N*K*sizeof(int8_t), &weights_dma, &weights_obj, 0, &weights_handle);

  uint64_t output_dma, output_obj;
  uint32_t output_handle;
  void *output = mem_allocate(fd, M*N*sizeof(int32_t), &output_dma, &output_obj, 0, &output_handle);

  printf("input dma is %lx, output dma is %lx, weights dma is %lx\n", input_dma, output_dma, weights_dma);
  if ((regcmd == NULL) || (tasks == NULL) || (input == NULL) || (weights == NULL) || (output == NULL)) {
    printf("Failed to allocate memory \n");
    exit(1);
  }

  // Reset the NPU
  npu_reset(fd);

  matmul_params_t params;
  params.m = M;
  params.k = K;
  params.n = N;
  params.input_dma = input_dma;
  params.weights_dma = weights_dma;
  params.output_dma = output_dma;
  params.tasks = (uint64_t *)&npu_regs;
  ret = gen_matmul_int8(&params);
  if (ret !=0) {
    printf("gen_matmul_int8 failed %d\n",ret);
    goto cleanup;
  }

  memcpy(regcmd,npu_regs,sizeof(npu_regs));

  tasks[0].flags  = 0;
  tasks[0].op_idx = 0;
  tasks[0].enable_mask = 0xd;
  tasks[0].int_mask = 0x300; // wait for DPU to finish
  tasks[0].int_clear = 0x1ffff;
  tasks[0].int_status = 0;
  tasks[0].regcfg_amount = sizeof(npu_regs)/sizeof(uint64_t)-(RKNPU_PC_DATA_EXTRA_AMOUNT+4);
  tasks[0].regcfg_offset = 0;
  tasks[0].regcmd_addr = regcmd_dma;

  memset((void *)input,0,M*K*sizeof(int8_t));
  memset((void *)weights,0,K*N*sizeof(int8_t));
  memset((void *)output,0,M*N*sizeof(int32_t));

  srand(time(NULL));

  for (int i = 0; i < M*K; i++) {
    matrixA[i] = rand_int();
  }

  for (int i = 0; i < N*K; i++) {
    matrixB[i] = rand_int();
  }
  
  int8_t *weights_int8 = weights;

  for(int n=1;n<=N;n++) {
    for(int k=1;k<=K;k++) {
      weights_int8[weight_int8(K,n,k)]= matrixB[((n-1)*K)+(k-1)];
    }
  }
 
  int8_t *feature_data_int8 = (int8_t*) input;

  for (int m=1;m<=M;m++) {
    for (int k=1;k<=K;k++) {
      feature_data_int8[feature_data(K,M,1,16,k,m,1)]= matrixA[((m-1)*K)+(k-1)];
    }
  }

  matmul_int(M,K,N,(int8_t *)&matrixA, (int8_t *)&matrixB, (int32_t *)&expected_result);

  struct rknpu_submit submit = {
    .flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
    .timeout = 6000,
    .task_start = 0,
    .task_number = 1,
    .task_counter = 0,
    .priority = 0,
    .task_obj_addr = tasks_obj,
    .regcfg_obj_addr = 0,
    .task_base_addr = 0,
    .user_data = 0,
    .core_mask = 1,
    .fence_fd = -1,
    .subcore_task = { // Only use core 1, nothing for core 2/3
      {
        .task_start = 0,
        .task_number = 1,
      }, { 1, 0}, {2, 0}, {0,0}, {0,0}
    },
  };
  ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);
  printf("RKNPU_SUBMIT returned %d\n", ret);
  if (ret <0)  {
    return ret;
  }

  printf("=========================================================================================================\n");
  int32_t *output_data = (int32_t*) output;
  for (int m=1;m<=M;m++) {
    for (int n=1;n<N;n++) {
      int32_t actual = output_data[feature_data(N, M, 1, 4, n, m, 1)];
      int32_t expected = expected_result[((m-1)*N)+(n-1)];
      if (actual != expected) {
        printf("\nmismatch m:%d n:%d  expected:%d acutal:%d \n",m,n,expected,actual);
        ret = -1;
      }
    }
  }
  if (ret == 0) {
   printf("Multiplication of [%d,%d] x [%d,%d] succesful \n",M,K,N,K);	  
  }
  printf("=========================================================================================================\n");

cleanup:
  munmap(regcmd,1024);
  munmap(tasks,1024);
  munmap(input,M*K*sizeof(int8_t));
  munmap(weights,N*K*sizeof(int8_t));
  munmap(output,M*N*sizeof(int32_t));

  mem_destroy(fd, regcmd_handle, regcmd_obj);
  mem_destroy(fd, tasks_handle, tasks_obj );
  mem_destroy(fd, input_handle, input_obj);
  mem_destroy(fd, weights_handle, weights_obj);
  mem_destroy(fd, output_handle, output_obj);

  npu_close(fd);
  return ret;
}
