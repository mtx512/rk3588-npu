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

#include <stdint.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include "rknpu-ioctl.h"
#include "npu_hw.h"

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

int npu_open() {

  char buf1[256], buf2[256], buf3[256];

  memset(buf1, 0 ,sizeof(buf1));
  memset(buf2, 0 ,sizeof(buf2));
  memset(buf3, 0, sizeof(buf3));

  // Open DRI called "rknpu"
  int fd = open("/dev/dri/card1", O_RDWR);
  if(fd<0) {
    printf("Failed to open /dev/dri/card1 %d\n",errno);
    return fd;
  }

  struct drm_version dv;
  memset(&dv, 0, sizeof(dv));
  dv.name = buf1;
  dv.name_len = sizeof(buf1);
  dv.date = buf2;
  dv.date_len = sizeof(buf2);
  dv.desc = buf3;
  dv.desc_len = sizeof(buf3);

  int ret = ioctl(fd, DRM_IOCTL_VERSION, &dv);
  if (ret <0) {
    printf("DRM_IOCTL_VERISON failed %d\n",ret);
    return ret;
  }
  printf("drm name is %s - %s - %s\n", dv.name, dv.date, dv.desc);
  return fd;
}

int npu_close(int fd) {
  return close(fd);	
}

int npu_reset(int fd) {

  // Reset the NPU
  struct rknpu_action act = {
    .flags = RKNPU_ACT_RESET,
  };
  return ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);	
}
