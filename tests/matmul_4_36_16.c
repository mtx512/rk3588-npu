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
#include <sys/mman.h>

#include <libdrm/drm.h>

#include "rknpu-ioctl.h"

  // Test currently runs against kernel 5.10 haven't tested 6.1 kernel.

  // Test data is from test-mul-mat.cpp see https://github.com/ggerganov/ggml
  // matrix A (4 X 36)
  float matrixA[] = {
    2.0f, 9.0f, 2.0f, 10.0f, 6.0f, 4.0f, 3.0f, 6.0f, 3.0f, 6.0f, 9.0f, 7.0f, 8.0f, 8.0f, 3.0f, 3.0f, 10.0f, 5.0f, 2.0f, 10.0f, 7.0f, 10.0f, 9.0f, 3.0f, 6.0f, 6.0f, 5.0f, 10.0f, 2.0f, 3.0f, 6.0f, 1.0f, 9.0f, 4.0f, 10.0f, 4.0f,
    10.0f, 7.0f, 8.0f, 10.0f, 10.0f, 8.0f, 7.0f, 10.0f, 4.0f, 6.0f, 8.0f, 7.0f, 7.0f, 6.0f, 9.0f, 3.0f, 6.0f, 5.0f, 5.0f, 2.0f, 7.0f, 2.0f, 7.0f, 4.0f, 4.0f, 6.0f, 6.0f, 4.0f, 3.0f, 9.0f, 3.0f, 6.0f, 4.0f, 7.0f, 2.0f, 9.0f,
    7.0f, 3.0f, 2.0f, 5.0f, 7.0f, 3.0f, 10.0f, 2.0f, 6.0f, 1.0f, 4.0f, 7.0f, 5.0f, 10.0f, 3.0f, 10.0f, 4.0f, 5.0f, 5.0f, 1.0f, 6.0f, 10.0f, 7.0f, 4.0f, 5.0f, 3.0f, 9.0f, 9.0f, 8.0f, 6.0f, 9.0f, 2.0f, 3.0f, 6.0f, 8.0f, 5.0f,
    5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 10.0f, 4.0f, 1.0f, 8.0f, 8.0f, 9.0f, 8.0f, 4.0f, 1.0f, 4.0f, 9.0f, 3.0f, 6.0f, 3.0f, 1.0f, 4.0f, 8.0f, 3.0f, 10.0f, 8.0f, 6.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f,
    };

  // matrix B (16 X 36)
  float  matrixB[] = {
    9.0f, 7.0f, 1.0f, 3.0f, 5.0f, 9.0f, 7.0f, 6.0f, 1.0f, 10.0f, 1.0f, 1.0f, 7.0f, 2.0f, 4.0f, 9.0f, 10.0f, 4.0f, 5.0f, 5.0f, 7.0f, 1.0f, 7.0f, 7.0f, 2.0f, 9.0f, 5.0f, 10.0f, 7.0f, 4.0f, 8.0f, 9.0f, 9.0f, 3.0f, 10.0f, 2.0f,
    4.0f, 6.0f, 10.0f, 9.0f, 5.0f, 1.0f, 8.0f, 7.0f, 4.0f, 7.0f, 2.0f, 6.0f, 5.0f, 3.0f, 1.0f, 10.0f, 8.0f, 4.0f, 8.0f, 3.0f, 7.0f, 1.0f, 2.0f, 7.0f, 6.0f, 8.0f, 6.0f, 5.0f, 2.0f, 3.0f, 1.0f, 1.0f, 2.0f, 5.0f, 7.0f, 1.0f,
    8.0f, 2.0f, 8.0f, 8.0f, 8.0f, 8.0f, 4.0f, 4.0f, 6.0f, 10.0f, 10.0f, 9.0f, 2.0f, 9.0f, 3.0f, 7.0f, 7.0f, 1.0f, 4.0f, 9.0f, 1.0f, 2.0f, 3.0f, 6.0f, 1.0f, 10.0f, 5.0f, 8.0f, 9.0f, 4.0f, 6.0f, 2.0f, 3.0f, 1.0f, 2.0f, 7.0f,
    5.0f, 1.0f, 7.0f, 2.0f, 9.0f, 10.0f, 9.0f, 5.0f, 2.0f, 5.0f, 4.0f, 10.0f, 9.0f, 9.0f, 1.0f, 9.0f, 8.0f, 8.0f, 9.0f, 4.0f, 9.0f, 4.0f, 8.0f, 2.0f, 1.0f, 8.0f, 4.0f, 5.0f, 10.0f, 7.0f, 6.0f, 2.0f, 1.0f, 10.0f, 10.0f, 7.0f,
    9.0f, 4.0f, 5.0f, 9.0f, 5.0f, 10.0f, 10.0f, 3.0f, 6.0f, 6.0f, 4.0f, 4.0f, 4.0f, 8.0f, 5.0f, 4.0f, 9.0f, 1.0f, 9.0f, 9.0f, 1.0f, 7.0f, 9.0f, 2.0f, 10.0f, 9.0f, 10.0f, 8.0f, 3.0f, 3.0f, 9.0f, 3.0f, 9.0f, 10.0f, 1.0f, 8.0f,
    9.0f, 2.0f, 6.0f, 9.0f, 7.0f, 2.0f, 3.0f, 5.0f, 3.0f, 6.0f, 9.0f, 7.0f, 3.0f, 7.0f, 6.0f, 4.0f, 10.0f, 3.0f, 5.0f, 7.0f, 2.0f, 9.0f, 3.0f, 2.0f, 2.0f, 10.0f, 8.0f, 7.0f, 3.0f, 10.0f, 6.0f, 3.0f, 1.0f, 1.0f, 4.0f, 10.0f,
    2.0f, 9.0f, 2.0f, 10.0f, 6.0f, 4.0f, 3.0f, 6.0f, 3.0f, 6.0f, 9.0f, 7.0f, 8.0f, 8.0f, 3.0f, 3.0f, 10.0f, 5.0f, 2.0f, 10.0f, 7.0f, 10.0f, 9.0f, 3.0f, 6.0f, 6.0f, 5.0f, 10.0f, 2.0f, 3.0f, 6.0f, 1.0f, 9.0f, 4.0f, 10.0f, 4.0f,
    10.0f, 7.0f, 8.0f, 10.0f, 10.0f, 8.0f, 7.0f, 10.0f, 4.0f, 6.0f, 8.0f, 7.0f, 7.0f, 6.0f, 9.0f, 3.0f, 6.0f, 5.0f, 5.0f, 2.0f, 7.0f, 2.0f, 7.0f, 4.0f, 4.0f, 6.0f, 6.0f, 4.0f, 3.0f, 9.0f, 3.0f, 6.0f, 4.0f, 7.0f, 2.0f, 9.0f,
    7.0f, 3.0f, 2.0f, 5.0f, 7.0f, 3.0f, 10.0f, 2.0f, 6.0f, 1.0f, 4.0f, 7.0f, 5.0f, 10.0f, 3.0f, 10.0f, 4.0f, 5.0f, 5.0f, 1.0f, 6.0f, 10.0f, 7.0f, 4.0f, 5.0f, 3.0f, 9.0f, 9.0f, 8.0f, 6.0f, 9.0f, 2.0f, 3.0f, 6.0f, 8.0f, 5.0f,
    5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 10.0f, 4.0f, 1.0f, 8.0f, 8.0f, 9.0f, 8.0f, 4.0f, 1.0f, 4.0f, 9.0f, 3.0f, 6.0f, 3.0f, 1.0f, 4.0f, 8.0f, 3.0f, 10.0f, 8.0f, 6.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f,
    6.0f, 2.0f, 3.0f, 3.0f, 3.0f, 7.0f, 5.0f, 1.0f, 8.0f, 1.0f, 4.0f, 5.0f, 1.0f, 1.0f, 6.0f, 4.0f, 2.0f, 1.0f, 7.0f, 8.0f, 6.0f, 1.0f, 1.0f, 5.0f, 6.0f, 5.0f, 10.0f, 6.0f, 7.0f, 5.0f, 9.0f, 3.0f, 2.0f, 7.0f, 9.0f, 4.0f,
    2.0f, 5.0f, 9.0f, 5.0f, 10.0f, 3.0f, 1.0f, 8.0f, 1.0f, 7.0f, 1.0f, 8.0f, 1.0f, 6.0f, 7.0f, 8.0f, 4.0f, 9.0f, 5.0f, 10.0f, 3.0f, 7.0f, 6.0f, 8.0f, 8.0f, 5.0f, 6.0f, 8.0f, 10.0f, 9.0f, 4.0f, 1.0f, 3.0f, 3.0f, 4.0f, 7.0f,
    8.0f, 2.0f, 6.0f, 6.0f, 5.0f, 1.0f, 3.0f, 7.0f, 1.0f, 7.0f, 2.0f, 2.0f, 2.0f, 8.0f, 4.0f, 1.0f, 1.0f, 5.0f, 9.0f, 4.0f, 1.0f, 2.0f, 3.0f, 10.0f, 1.0f, 4.0f, 9.0f, 9.0f, 6.0f, 8.0f, 8.0f, 1.0f, 9.0f, 10.0f, 4.0f, 1.0f,
    8.0f, 5.0f, 8.0f, 9.0f, 4.0f, 8.0f, 2.0f, 1.0f, 1.0f, 9.0f, 4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 3.0f, 1.0f, 4.0f, 6.0f, 7.0f, 7.0f, 7.0f, 8.0f, 7.0f, 8.0f, 8.0f, 2.0f, 10.0f, 2.0f, 7.0f, 3.0f, 8.0f, 3.0f,
    8.0f, 7.0f, 6.0f, 2.0f, 4.0f, 10.0f, 10.0f, 6.0f, 10.0f, 3.0f, 7.0f, 6.0f, 4.0f, 3.0f, 5.0f, 5.0f, 5.0f, 3.0f, 8.0f, 10.0f, 3.0f, 4.0f, 8.0f, 4.0f, 2.0f, 6.0f, 8.0f, 9.0f, 6.0f, 9.0f, 4.0f, 3.0f, 5.0f, 2.0f, 2.0f, 6.0f,
    10.0f, 6.0f, 2.0f, 1.0f, 7.0f, 5.0f, 6.0f, 4.0f, 1.0f, 9.0f, 10.0f, 2.0f, 4.0f, 5.0f, 8.0f, 5.0f, 7.0f, 4.0f, 7.0f, 6.0f, 3.0f, 9.0f, 2.0f, 1.0f, 4.0f, 2.0f, 6.0f, 6.0f, 3.0f, 3.0f, 2.0f, 8.0f, 5.0f, 9.0f, 3.0f, 4.0f,
    };

  // matrix C (4 x 16)
  float expected_result[] = {
    1224.0f, 1023.0f, 1158.0f,1259.0f,1359.0f,1194.0f,1535.0f,1247.0f,1185.0f,1029.0f,889.0f,1182.0f,955.0f,1179.0f,1147.0f,1048.0f,
    1216.0f, 1087.0f, 1239.0f,1361.0f,1392.0f,1260.0f,1247.0f,1563.0f,1167.0f,1052.0f,942.0f,1214.0f,1045.0f,1134.0f,1264.0f,1126.0f,
    1125.0f, 966.0f, 1079.0f,1333.0f,1287.0f,1101.0f,1185.0f,1167.0f,1368.0f,990.0f,967.0f,1121.0f,971.0f,1086.0f,1130.0f,980.0f,
    999.0f, 902.0f, 1020.0f,1056.0f,1076.0f,929.0f,1029.0f,1052.0f,990.0f,1108.0f,823.0f,989.0f,759.0f,1041.0f,1003.0f,870.0f
    };

  // Hand crafted register definitions for a simple fp 16 convolution which
  // can be done with single NPU task because the input cube and weights are
  // small. Feature data is 4x1x40 and weights 1x1x40x16, output is 4x1x16.
  // Note: numerous registers require changes if the input cube or weight
  // dimensions are altered.
 
  uint64_t npu_regs[] = {
    0x10010000000e4004, 0x020100000120100c, 0x0201000000101010, 0x0201000000091014,
    0x0201000100041020, 0x0201002700281024, 0x0201000000011028, 0x020100000004102c,
    0x0201000005001030, 0x0201000000501034, 0x0201010100101038, 0x02010000001b1040,
    0x0201000000021044, 0x020100000001104c, 0x0201000100001050, 0x0201000100001054,
    0x0201000100001058, 0x020100010000105c, 0x0201000000001060, 0x0201000000001064,
    0x0201000000001068, 0x0201000000001070, 0x0201000000001074, 0x0201000f000f1078,
    0x020100000010107c, 0x0201000000001080, 0x0201000100041084, 0x0201000000281088,
    0x0201000000001100, 0x0201000000001104, 0x0201000000001110, 0x0201000000001140,
    0x0201000000001144, 0x0201000000001148, 0x020100000000114c, 0x0201000000001150,
    0x0201000000001154, 0x0201000000001158, 0x020100000000115c, 0x0201000000001160,
    0x0201000000001164, 0x0201000000001168, 0x020100000000116c, 0x0201000000001170,
    0x0201000000001174, 0x0201000000001178, 0x020100000000117c, 0x0201000000001180,
    0x0201ffffff801184, 0x0801000002013010, 0x0801000300003014, 0x0801000000103018,
    0x080100000000301c, 0x0801000000003030, 0x1001000001e4400c, 0x1001a80000054010,
    0x1001000000004014, 0x1001000000004020, 0x1001000000404024, 0x1001000000004030,
    0x1001000000034034, 0x1001000000004038, 0x1001000f000f403c, 0x1001000000534040,
    0x1001000000004044, 0x1001000000004048, 0x100100000000404c, 0x10010000036e4050,
    0x1001000000004054, 0x10010000000f4058, 0x100100030000405c, 0x1001000000534060,
    0x1001000000004064, 0x1001000000004068, 0x100100000000406c, 0x1001000003834070,
    0x1001000000004074, 0x1001000000004078, 0x100100000000407c, 0x1001000000004080,
    0x1001000000004084, 0x1001000000004088, 0x1001000000004090, 0x1001000000004094,
    0x1001000000004098, 0x100100000000409c, 0x10010000000040a0, 0x10010000000040a4,
    0x10010000000040a8, 0x10010000000040ac, 0x10010000010040c0, 0x10010000000040c4,
    0x1001000000004100, 0x1001000000004104, 0x1001000000004108, 0x100100000000410c,
    0x1001000000004110, 0x1001000000004114, 0x1001000000004118, 0x100100000000411c,
    0x1001000000004120, 0x1001000000004124, 0x1001000000004128, 0x100100000000412c,
    0x0101000000000000, 0x0101000000000014, 0x0041000000000000, 0x00810000000d0008,
    // Single task needs at least 112 entries therefore pad it out
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
  };

int feature_data(int C, int H, int W, int C2, int c, int h, int w) {

  int plane = (c-1)/C2;
  int src = plane * H * W * C2;
  int offset = (c-1) % C2;
  int pos = src + C2 * ((h-1) * W + (w-1)) + offset;
  return pos;
}


int weight_data(int K, int C, int k, int c) {

  // fp16 format

  int cpg=32;  
  int kgs = (C/cpg)+1;  
  int gi = ((c-1)/cpg)+1;
  int C2_gs = ((C-1)/8)+1;
  int c2_gs = ((c-1)/8)+1;
  int c1_gs = ((c2_gs-1)/4);
  int dst = c1_gs * 32 * K;
  int rgs = (C2_gs)-(c1_gs*4);
  int r=(c-1)%cpg;
  if (gi == kgs) {
    dst = dst + (rgs*8*(k-1));
  } else {
    dst = dst + (cpg*(k-1));
  }
  dst = dst + r;
  return dst;
}	

void* mem_allocate(int fd, size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags, uint32_t *handle) {

  int ret;
  struct rknpu_mem_create mem_create = {
    .flags = flags | RKNPU_MEM_NON_CACHEABLE,
    .size = size,
  };

  ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create);
  if(ret < 0)  {
    printf("RKNPU_MEM_CREATE failed %d\n",ret);
    return NULL;
  }

  struct rknpu_mem_map mem_map = { .handle = mem_create.handle, .offset=0 };
  ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mem_map);
  if(ret < 0) {
    printf("RKNPU_MEM_MAP failed %d\n",ret);
    return NULL;
  }	
  void *map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mem_map.offset);

  *dma_addr = mem_create.dma_addr;
  *obj = mem_create.obj_addr;
  *handle = mem_create.handle;
  return map;
}

void mem_destroy(int fd, uint32_t handle, uint64_t obj_addr) {

  int ret;
  struct rknpu_mem_destroy destroy = {
    .handle = handle ,
    .obj_addr = obj_addr
  };

  ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &destroy);
  if (ret <0) {
    printf("RKNPU_MEM_DESTROY failed %d\n",ret);
  }
}

int main(int argc, char **argv) {

  char buf1[256], buf2[256], buf3[256];
  memset(buf1, 0, sizeof(buf1));
  memset(buf2, 0, sizeof(buf2));
  memset(buf3, 0, sizeof(buf3));

  int ret;
  int M=4;
  int K=36;
  int N=16;

  // Open DRI called "rknpu"
  int fd = open("/dev/dri/card1", O_RDWR);
  if(fd<0) {
    printf("Failed to open /dev/dri/card1 %d\n",errno);
    exit(1);
  }

  struct drm_version dv;
  memset(&dv, 0, sizeof(dv));
  dv.name = buf1;
  dv.name_len = sizeof(buf1);
  dv.date = buf2;
  dv.date_len = sizeof(buf2);
  dv.desc = buf3;
  dv.desc_len = sizeof(buf3);

  ret = ioctl(fd, DRM_IOCTL_VERSION, &dv);
  if (ret <0) {
    printf("DRM_IOCTL_VERISON failed %d\n",ret);
    exit(1);
  }
  printf("drm name is %s - %s - %s\n", dv.name, dv.date, dv.desc);

  uint64_t regcmd_dma, regcmd_obj;
  uint32_t regcmd_handle;
  uint64_t *regcmd = mem_allocate(fd, 1024, &regcmd_dma, &regcmd_obj, 0, &regcmd_handle);

  uint64_t tasks_dma, tasks_obj;
  uint32_t tasks_handle;
  struct rknpu_task *tasks = mem_allocate(fd, 1024, &tasks_dma, &tasks_obj, RKNPU_MEM_KERNEL_MAPPING, &tasks_handle);

  uint64_t input_dma, input_obj;
  uint32_t input_handle;
  void *input = mem_allocate(fd, 4096, &input_dma, &input_obj, 0, &input_handle);

  uint64_t weights_dma, weights_obj;
  uint32_t weights_handle;
  void *weights = mem_allocate(fd, 4096, &weights_dma, &weights_obj, 0, &weights_handle);

  uint64_t output_dma, output_obj;
  uint32_t output_handle;
  void *output = mem_allocate(fd, 4096, &output_dma, &output_obj, 0, &output_handle);

  printf("input dma is %lx, output dma is %lx, weights dma is %lx\n", input_dma, output_dma, weights_dma);
  if ((regcmd == NULL) || (tasks == NULL) || (input == NULL) || (weights == NULL) || (output == NULL)) {
    printf("Failed to allocate memory \n");
    exit(1);
  }

  // Reset the NPU
  struct rknpu_action act = {
    .flags = RKNPU_ACT_RESET,
  };
  ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);

  // Set input, weights and output physical memory locations. Note limited to 
  // a 32 bit address size (4GB)
  npu_regs[21] = npu_regs[21] | ((input_dma & 0xFFFFFFFF) <<16);
  npu_regs[30] = npu_regs[30] | ((weights_dma & 0xFFFFFFFF)  <<16);
  npu_regs[57] = npu_regs[57] | ((output_dma & 0xFFFFFFFF) <<16);
  printf("Size of npu_regs %d\n",(int)(sizeof(npu_regs)/sizeof(uint64_t)));
  memcpy(regcmd,npu_regs,sizeof(npu_regs));

  tasks[0].flags  = 0;
  tasks[0].op_idx = 1;
  tasks[0].enable_mask = 0x7f;
  tasks[0].int_mask = 0x300; // wait for DPU to finish
  tasks[0].int_clear = 0x1ffff;
  tasks[0].regcfg_amount = sizeof(npu_regs)/sizeof(uint64_t); //nInstrs - 1;
  tasks[0].regcfg_offset = 0;
  tasks[0].regcmd_addr = regcmd_dma;

  memset((void *)input,0,M*K*sizeof(__fp16));
  memset((void *)weights,0,K*N*sizeof(__fp16));
  memset((void *)output,0,M*N*sizeof(float));

  __fp16 *weights_fp16 = weights;
   
  for(int n=1;n<=N;n++) {
    for(int k=1;k<=K;k++) {
      weights_fp16[weight_data(N,40,n,k)]= matrixB[((n-1)*K)+(k-1)];
    }
  }

  __fp16 *feature_data_fp16 = (__fp16*) input;

  for (int m=1;m<=M;m++) {
    for (int k=1;k<=K;k++) {
      feature_data_fp16[feature_data(40,4,1,8,k,m,1)]= matrixA[((m-1)*K)+(k-1)];
    }
  }

  munmap(input,4096);
  munmap(weights,4096);

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
      }, { 1, 0}, {2, 0},
    },
  };
  ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);
  printf("RKNPU_SUBMIT returned %d\n", ret);

  printf("=========================================================================================================\n");
  float *output_data = (float*) output;

  for (int m=1;m<=M;m++) {
    for (int n=1;n<N;n++) {
      float actual = output_data[feature_data(N, 4, 1, 4, n, m, 1)];
      if (actual == expected_result[((m-1)*N)+(n-1)]) {
        printf("\033[0;32m %6.1f",actual);
      } else {
        printf("\033[0;31m %6.1f",actual);
      }
    }
    printf("\n");
  }
  printf("\033[0;37m");
  printf("=========================================================================================================\n");

  munmap(regcmd,1024);
  munmap(tasks,1024);
  munmap(input,4096);
  munmap(weights,4096);
  munmap(output,4096);

  mem_destroy(fd, regcmd_handle, regcmd_obj);
  mem_destroy(fd, tasks_handle, tasks_obj );
  mem_destroy(fd, input_handle, input_obj);
  mem_destroy(fd, weights_handle, weights_obj);
  mem_destroy(fd, output_handle, output_obj);

  close(fd);
  return 0;
}
