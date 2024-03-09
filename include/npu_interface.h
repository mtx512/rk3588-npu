#ifndef NPU_INTERFACE_H
#define NPU_INTERFACE_H

#include <stdint.h>

void* mem_allocate(int fd, size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags, uint32_t *handle);
void mem_destroy(int fd, uint32_t handle, uint64_t obj_addr);

int npu_open();
int npu_close(int fd);
int npu_reset(int fd);

#endif // NPU_INTERFACE_H
