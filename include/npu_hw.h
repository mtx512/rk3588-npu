#ifndef NPU_HW_H
#define NPU_HW_H

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

// Registers as per TRM V1.0 2022-03-09 and descriptions (can be cryptic or missing)

#define PC_OPERATION_ENABLE    0x0008 // Operation Enable
#define PC_BASE_ADDRESS        0x0010 // PC address register
#define PC_REGISTER_AMOUNTS    0x0014 // Register amount for each task

#define CNA_S_POINTER          0x1004 // Single register group pointer
#define CNA_CONV_CON1          0x100C // Convolution control register1
#define CNA_CONV_CON2          0x1010 // Convolution control register2
#define CNA_CONV_CON3          0x1014 // Convolution control register3
#define CNA_DATA_SIZE0         0x1020 // Feature data size control register0
#define CNA_DATA_SIZE1         0x1024 // Feature data size control register1
#define CNA_DATA_SIZE2         0x1028 // Feature data size control register2
#define CNA_DATA_SIZE3         0x102C // Feature data size control register3
#define CNA_WEIGHT_SIZE0       0x1030 // Weight size control 0
#define CNA_WEIGHT_SIZE1       0x1034 // Weight size control 1
#define CNA_WEIGHT_SIZE2       0x1038 // Weight size control 2
#define CNA_CBUF_CON0          0x1040 // CBUF control register 0
#define CNA_CBUF_CON1          0x1044 // CBUF control register 1
#define CNA_CVT_CON0           0x104C // Input convert control register0
#define CNA_CVT_CON1           0x1050 // Input convert control register1
#define CNA_CVT_CON2           0x1054 // Input convert control register2
#define CNA_CVT_CON3           0x1058 // Input convert control register3
#define CNA_CVT_CON4           0x105C // Input convert control register4
#define CNA_FC_CON0            0x1060 // Full connected control register0
#define CNA_FC_CON1            0x1064 // Full connected control register1
#define CNA_PAD_CON0           0x1068 // Pad control register0
#define CNA_FEATURE_DATA_ADDR  0x1070 // Base address for input feature data
#define CNA_FC_CON2            0x1074 // Full connected control register2
#define CNA_DMA_CON0           0x1078 // AXI control register 0
#define CNA_DMA_CON1           0x107C // AXI control register 1
#define CNA_DMA_CON2           0x1080 // AXI control register 2
#define CNA_FC_DATA_SIZE0      0x1084 // Full connected data size control register0
#define CNA_FC_DATA_SIZE1      0x1088 // Full connected data size control register1
#define CNA_DCOMP_CTRL         0x1100 // Weight decompress control register
#define CNA_DCOMP_REGNUM       0x1104 // Weight decompress register number
#define CNA_DCOMP_ADDR0        0x1110 // Base address of the weight
#define CNA_DCOMP_AMOUNT       0x1140 // Amount of the weight decompress for the 0 decompress
#define CNA_DCOMP_AMOUNT1      0x1144 // Amount of the weight decompress for the 1 decompress
#define CNA_DCOMP_AMOUNT2      0x1148 // Amount of the weight decompress for the 2 decompress
#define CNA_DCOMP_AMOUNT3      0x114C // Amount of the weight decompress for the 3 decompress
#define CNA_DCOMP_AMOUNT4      0x1150 // Amount of the weight decompress for the 4 decompress
#define CNA_DCOMP_AMOUNT5      0x1154 // Amount of the weight decompress for the 5 decompress
#define CNA_DCOMP_AMOUNT6      0x1158 // Amount of the weight decompress for the 6 decompress
#define CNA_DCOMP_AMOUNT7      0x115C // Amount of the weight decompress for the 7 decompress
#define CNA_DCOMP_AMOUNT8      0x1160 // Amount of the weight decompress for the 8 decompress
#define CNA_DCOMP_AMOUNT9      0x1164 // Amount of the weight decompress for the 9 decompress
#define CNA_DCOMP_AMOUNT10     0x1168 // Amount of the weight decompress for the 10 decompress
#define CNA_DCOMP_AMOUNT11     0x116C // Amount of the weight decompress for the 11 decompress
#define CNA_DCOMP_AMOUNT12     0x1170 // Amount of the weight decompress for the 12 decompress
#define CNA_DCOMP_AMOUNT13     0x1174 // Amount of the weight decompress for the 13 decompress
#define CNA_DCOMP_AMOUNT14     0x1178 // Amount of the weight decompress for the 14 decompress
#define CNA_DCOMP_AMOUNT15     0x117C // Amount of the weight decompress for the 15 decompress
#define CNA_CVT_CON5           0x1180 // Input convert control register5
#define CNA_PAD_CON1           0x1184 // Pad controller register1

#define CORE_S_POINTER         0x3004 // Single register group pointer
#define CORE_MISC_CFG          0x3010 // Misc configuration register
#define CORE_DATAOUT_SIZE_0    0x3014 // Feature size register 0 of output
#define CORE_DATAOUT_SIZE_1    0x3018 // Feature size register 1 of output
#define CORE_CLIP_TRUNCATE     0x301C // Shift value register
#define CORE_3030              0x3030 // Doesn't seem to be documented, is it required ??

#define DPU_S_POINTER            0x4004 // Single register group pointer
#define DPU_FEATURE_MODE_CFG     0x400C // Configuration of the feature mode
#define DPU_DATA_FORMAT          0x4010 // Configuration of the data format
#define DPU_OFFSET_PEND          0x4014 // Value of the offset pend
#define DPU_DST_BASE_ADD         0x4020 // Destination base address
#define DPU_DST_SURF_STRIDE      0x4024 // Destination surface size
#define DPU_DATA_CUBE_WIDTH      0x4030 // Width of the input cube
#define DPU_DATA_CUBE_HEIGHT     0x4034 // Height of the input cube  
#define DPU_DATA_CUBE_NOTCH_ADDR 0x4038 // Notch signal of the input cube
#define DPU_DATA_CUBE_CHANNEL    0x403C // Channel of the input cube
#define DPU_BS_CFG               0x4040 // Configuration of the BS
#define DPU_BS_ALU_CFG           0x4044 // Configuration of the BS ALU
#define DPU_BS_MUL_CFG           0x4048 // Configuration of the BS MUL
#define DPU_BS_RELUX_CMP_VALUE   0x404C // Value of the RELUX compare with
#define DPU_BS_OW_CFG            0x4050 // Configuration of the BS OW
#define DPU_BS_OW_OP             0x4054 // Ow op of the BS OW
#define DPU_WDMA_SIZE_0          0x4058 // Size 0 of the WDMA
#define DPU_WDMA_SIZE_1          0x405C // Size 1 of the WDMA
#define DPU_BN_CFG               0x4060 // Configuration of BN
#define DPU_BN_ALU_CFG           0x4064 // Configuration of the BN ALU
#define DPU_BN_MUL_CFG           0x4068 // Configuration of the BN MUL
#define DPU_BN_RELUX_CMP_VALUE   0x406C // Value of the RELUX compare with
#define DPU_EW_CFG               0x4070 // Configuration of EW
#define DPU_EW_CVT_OFFSET_VALUE  0x4074 // Offset of the EW input convert
#define DPU_EW_CVT_SCALE_VALUE   0x4078 // Scale of the EW input convert
#define DPU_EW_RELUX_CMP_VALUE   0x407C // Value of the RELUX compare with
#define DPU_OUT_CVT_OFFSET       0x4080 // Offset of the output converter
#define DPU_OUT_CVT_SCALE        0x4084 // Scale of the output converter
#define DPU_OUT_CVT_SHIFT        0x4088 // Shift of the output converter
#define DPU_EW_OP_VALUE_0        0x4090 // Configure operand0 of the EW
#define DPU_EW_OP_VALUE_1        0x4094 // Configure operand1 of the EW
#define DPU_EW_OP_VALUE_2        0x4098 // Configure operand2 of the EW
#define DPU_EW_OP_VALUE_3        0x409C // Configure operand3 of the EW
#define DPU_EW_OP_VALUE_4        0x40A0 // Configure operand4 of the EW
#define DPU_EW_OP_VALUE_5        0x40A4 // Configure operand5 of the EW
#define DPU_EW_OP_VALUE_6        0x40A8 // Configure operand6 of the EW
#define DPU_EW_OP_VALUE_7        0x40AC // Configure operand7 of the EW
#define DPU_SURFACE_ADD          0x40C0 // Value of the surface adder
#define DPU_40C4                 0x40C4 // Not documented      
#define DPU_LUT_ACCESS_CFG       0x4100 // LUT access address and type
#define DPU_LUT_ACCESS_DATA      0x4104 // Configuration of LUT access data
#define DPU_LUT_CFG              0x4108 // Configuration of the LUT
#define DPU_LUT_INFO             0x410C // LUT information register
#define DPU_LUT_LE_START         0x4110 // LE LUT start point
#define DPU_LUT_LE_END           0x4114 // LE LUT end point
#define DPU_LUT_LO_START         0x4118 // LO LUT start point
#define DPU_LUT_LO_END           0x411C // LO LUT end point
#define DPU_LUT_LE_SLOPE_SCALE   0x4120 // LE LUT slope scale
#define DPU_LUT_LE_SLOPE_SHIFT   0x4124 // LE LUT slope shift
#define DPU_LUT_LO_SLOPE_SCALE   0x4128 // LO LUT slope scale
#define DPU_LUT_LO_SLOPE_SHIFT   0x412C // LO LUT slope shift

// TODO Add PPU

// NPU capability is limited to the following units
#define BLOCK_PC       0x0100
#define BLOCK_CNA      0x0200
#define BLOCK_CORE     0x0800
#define BLOCK_DPU      0x1000
#define BLOCK_DPU_RDMA 0x2000
#define BLOCK_PPU      0x4000
#define BLOCK_PPU_RDMA 0x8000

#define PC_OP_01     0x01  // reg ??
#define PC_OP_40     0x40  // ??
#define PC_OP_ENABLE 0x80  // Enables block(s)  

#define OP_REG_PC   (BLOCK_PC | PC_OP_01)   // ??
#define OP_REG_CNA  (BLOCK_CNA | PC_OP_01)  // ??
#define OP_REG_CORE (BLOCK_CORE | PC_OP_01) // ??
#define OP_REG_DPU  (BLOCK_DPU | PC_OP_01)  // ??

#define OP_40     (PC_OP_40 | PC_OP_01)     // ??
#define OP_ENABLE (PC_OP_ENABLE | PC_OP_01) // ??
#define OP_NONE   0x0                       // ??

#define PC_ENABLE      0x01  // Enable for this task
#define PC_ENABLE_CNA  0x04  // ?? Interrupt
#define PC_ENABLE_DPU  0x08  // ?? Interrupt
#define PC_ENABLE_PPU  0x10  // ?? Interrupt

#define NPUOP(op, value, reg) (((uint64_t)(op & 0xffff))<< 48) | ( ((uint64_t)(value & 0xffffffff)) << 16) | (uint64_t)(reg & 0xffff)

#define NPU_CBUF_BANK_SIZE 32768
#define NPU_CBUF_BANKS 12

enum  { direct_convolution = 0}; 
enum  { precision_int8 = 0,
        precision_float16 = 2,
        precision_int32 = 4,
        precision_float32 = 5};

#endif //NPU_HW_H
