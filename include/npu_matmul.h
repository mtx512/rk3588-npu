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

typedef struct {
  uint16_t  m;
  uint16_t  k;
  uint16_t  n;

  uint32_t  input_dma;
  uint32_t  weights_dma;
  uint32_t  output_dma;

  uint64_t  *tasks;
} matmul_params_t;

int gen_matmul_fp16(matmul_params_t *params);
int gen_matmul_int8(matmul_params_t *params);
int feature_data(int C, int H, int W, int C2, int c, int h, int w);
int weight_fp16(int C, int k, int c);
int weight_int8(int C, int k, int c);

#endif // NPU_MATMUL_H
