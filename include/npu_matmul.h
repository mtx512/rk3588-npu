#ifndef NPU_MATMUL_H
#define NPU_MATMUL_H

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

#include "npu_cna.h"
#include "npu_dpu.h"

int gen_matmul_fp16(uint16_t M, uint16_t K, uint16_t N, uint32_t input, uint32_t weights, uint32_t output, uint64_t *task);
int gen_matmul_int8(uint16_t M, uint16_t K, uint16_t N, uint32_t input, uint32_t weights, uint32_t output, uint64_t *task);
int feature_data(int C, int H, int W, int C2, int c, int h, int w);
int weight_fp16(int C, int k, int c);
int weight_int8(int C, int k, int c);

#endif // NPU_MATMUL_H
