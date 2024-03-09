#ifndef NPU_MATMUL_H
#define NPU_MATMUL_H

#include "npu_cna.h"
#include "npu_dpu.h"

int gen_matmul_fp16(uint16_t M, uint16_t K, uint16_t N, uint32_t input, uint32_t weights, uint32_t output, uint64_t *task);

#endif // NPU_MATMUL_H
