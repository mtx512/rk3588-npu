#ifndef NPU_CNA_H
#define NPU_CNA_H

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

typedef struct npu_cna_desc {
  uint8_t enable;
  uint8_t conv_mode;          // 0x100C
  uint8_t in_precision;       // 0x100C
  uint8_t proc_precision;     // 0x100C
  uint8_t kernel_groups;      // 0x1010
  uint16_t feature_grains;    // 0x1010
  uint8_t conv_y_stride;      // 0x1014
  uint8_t conv_x_stride;      // 0x1014
  uint16_t datain_width;      // 0x1020
  uint16_t datain_height;     // 0x1020
  uint16_t datain_channel;    // 0x1024
  uint16_t dataout_width;     // 0x1028
  uint32_t dataout_atomics;   // 0x102C
  uint32_t weight_bytes;      // 0x1030
  uint32_t weight_bytes_per_kernel; // 0x1034
  uint8_t weight_width;       // 0x1038
  uint8_t weight_height;      // 0x1038
  uint16_t weight_kernels;    // 0x1038
  uint8_t weight_bank;        // 0x1040
  uint8_t data_bank;          // 0x1040
  uint16_t data_entries;      // 0x1044
  uint8_t data_sign;          // 0x104c
  uint8_t cvt_type;           // 0x104c
  uint8_t cvt_bypass;         // 0x104c
  uint16_t cvt_scale0;        // 0x1050
  uint16_t cvt_scale1;        // 0x1054
  uint16_t cvt_scale2;        // 0x1058
  uint16_t cvt_scale3;        // 0x105C
  uint8_t fc_skip_en;         // 0x1060
  uint16_t data_offset;       // 0x1064
  uint8_t pad_left;           // 0x1068
  uint8_t pad_top;            // 0x1068
  uint32_t feature_base_addr; // 0x1070
  uint16_t weight_offset;     // 0x1074
  uint8_t weight_burst_len;   // 0x1078
  uint8_t data_burst_len;     // 0x1078
  uint32_t line_stride;       // 0x107C
  int32_t surf_stride;        // 0x1080
  uint16_t dma_width;         // 0x1084
  uint16_t dma_height;        // 0x1084
  uint16_t dma_channel;       // 0x1088
  uint32_t decompress_addr0;  // 0x1110

  uint16_t dataout_height;
} npu_cna_desc;

typedef struct npu_core_desc {
  uint8_t proc_precision;   // 0x3010
  uint8_t qd_en;            // 0x3010
  uint16_t dataout_height;  // 0x3014
  uint16_t dataout_width;   // 0x3014
  uint16_t dataout_channel; // 0x3018
} npu_core_desc;

typedef struct nup_pc_desc {
 uint32_t pc_source_addr;  // 0x0010
 uint32_t pc_data_amount;  // 0x0014
} npu_pc_desc;

typedef struct npu_cna_core_task {

  // Use cna, core, bypass dpu operations & output to memory.
  // This can be done within 112 operations.
  uint64_t ops[112];

} npu_cna_core_task;

#endif //NPU_CNA_H
