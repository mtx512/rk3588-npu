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

#include <stddef.h>
#include <string.h>

#include <stdio.h>

#include "npu_hw.h"
#include "npu_cna.h"
#include "npu_dpu.h"

/*
 * Were only using cna & core, dpu outputs to memory
 *
 */
int gen_matmul_task(uint64_t *ops, npu_cna_desc *cna_desc, npu_core_desc *core_desc, npu_dpu_desc *dpu_desc) {

  uint32_t value;
  int ret =0;

  ops[0] = NPUOP(OP_REG_DPU, 0xE, DPU_S_POINTER);
  value = ((cna_desc->proc_precision & 0x7) <<7) |  ((cna_desc->in_precision & 0x7)<<4) | 
    (cna_desc->conv_mode & 0xf);
  ops[1] = NPUOP(OP_REG_CNA, value, CNA_CONV_CON1);
  value = ((cna_desc->feature_grains & 0x3FF) << 4);
  ops[2] = NPUOP(OP_REG_CNA, value, CNA_CONV_CON2);
  value = ((cna_desc->conv_y_stride & 0x7) << 3) | (cna_desc->conv_x_stride & 0x7);
  ops[3] = NPUOP(OP_REG_CNA, value, CNA_CONV_CON3);
  value = ((cna_desc->datain_width) & 0x7FF) << 16 | (cna_desc->datain_height & 0x7FF);
  ops[4] = NPUOP(OP_REG_CNA, value, CNA_DATA_SIZE0);
  value = ((cna_desc->datain_channel-1) & 0xFFFF) << 16 | (cna_desc->datain_channel & 0xFFFF);
  ops[5] = NPUOP(OP_REG_CNA, value, CNA_DATA_SIZE1);
  value = cna_desc->dataout_width & 0x7FF;
  ops[6] = NPUOP(OP_REG_CNA, value, CNA_DATA_SIZE2);
  value = cna_desc->dataout_atomics & 0x3FFFF;
  ops[7] = NPUOP(OP_REG_CNA, value, CNA_DATA_SIZE3);
  value = cna_desc->weight_bytes;
  ops[8] = NPUOP(OP_REG_CNA, value, CNA_WEIGHT_SIZE0);
  value = cna_desc->weight_bytes_per_kernel & 0x7FFFF;
  ops[9] = NPUOP(OP_REG_CNA, value, CNA_WEIGHT_SIZE1);
  value = ((cna_desc->weight_width & 0x1F) <<24) | ((cna_desc->weight_height & 0x1F) << 16) |
    (cna_desc->weight_kernels & 0x3FFF);
  ops[10] = NPUOP(OP_REG_CNA, value, CNA_WEIGHT_SIZE2);
  value = ((cna_desc->weight_bank & 0xF) << 4) | (cna_desc->data_bank & 0xF);
  ops[11] = NPUOP(OP_REG_CNA, value, CNA_CBUF_CON0);
  value = cna_desc->data_entries & 0x1FFF;
  ops[12] = NPUOP(OP_REG_CNA, value, CNA_CBUF_CON1);
  value = ((cna_desc->data_sign & 0x1) << 3) | (cna_desc->cvt_bypass);
  ops[13] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON0);
  value = ((cna_desc->cvt_scale0 & 0xFFFF) << 16) | 0x0;
  ops[14] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON1);
  value = ((cna_desc->cvt_scale1 & 0xFFFF) << 16) | 0x0;
  ops[15] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON2);
  value = ((cna_desc->cvt_scale2 & 0xFFFF) << 16) | 0x0;
  ops[16] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON3);
  value = ((cna_desc->cvt_scale3 & 0xFFFF) << 16) | 0x0;
  ops[17] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON4);
  value = cna_desc->fc_skip_en & 0x1;
  ops[18] = NPUOP(OP_REG_CNA, value, CNA_FC_CON0);
  value = cna_desc->data_offset & 0x1FFFF;
  ops[19] = NPUOP(OP_REG_CNA, value, CNA_FC_CON1); 
  value = ((cna_desc->pad_left & 0xF) << 4) | (cna_desc->pad_top & 0xF);
  ops[20] = NPUOP(OP_REG_CNA, value, CNA_PAD_CON0);
  ops[21] = NPUOP(OP_REG_CNA, cna_desc->feature_base_addr, CNA_FEATURE_DATA_ADDR);
  value = cna_desc->weight_offset & 0x1FFFF;
  ops[22] = NPUOP(OP_REG_CNA, value, CNA_FC_CON2);
  value = ((cna_desc->weight_burst_len & 0xF) << 16) | (cna_desc->data_burst_len & 0xF);
  ops[23] = NPUOP(OP_REG_CNA, value, CNA_DMA_CON0);
  value = cna_desc->line_stride & 0xFFFFFFF;
  ops[24] = NPUOP(OP_REG_CNA, value, CNA_DMA_CON1);
  value = cna_desc->surf_stride & 0xFFFFFFF;
  ops[25] = NPUOP(OP_REG_CNA, value, CNA_DMA_CON2);
  value = ((cna_desc->dma_width & 0x7FF) << 16) | (cna_desc->dma_height & 0x7FF);
  ops[26] = NPUOP(OP_REG_CNA, value, CNA_FC_DATA_SIZE0);
  value = cna_desc->dma_channel & 0xFFFF;
  ops[27] = NPUOP(OP_REG_CNA, value, CNA_FC_DATA_SIZE1);
  ops[28] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_CTRL);
  ops[29] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_REGNUM);
  ops[30] = NPUOP(OP_REG_CNA, cna_desc->decompress_addr0, CNA_DCOMP_ADDR0);
  ops[31] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT);
  ops[32] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT1);
  ops[33] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT2);
  ops[34] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT3);
  ops[35] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT4);
  ops[36] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT5);
  ops[37] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT6);
  ops[38] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT7);
  ops[39] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT8);
  ops[40] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT9);
  ops[41] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT10);
  ops[42] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT11);
  ops[43] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT12);
  ops[44] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT13);
  ops[45] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT14);
  ops[46] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT15);
  ops[47] = NPUOP(OP_REG_CNA, 0x0, CNA_CVT_CON5);
  ops[48] = NPUOP(OP_REG_CNA, 0x0, CNA_PAD_CON1);
  value = ((core_desc->proc_precision & 0x7) << 8) | (core_desc->qd_en & 0x1);
  ops[49] = NPUOP(OP_REG_CORE, value, CORE_MISC_CFG);
  value = ((core_desc->dataout_height & 0xFFFF) << 16) | (core_desc->dataout_width & 0xFFFF);
  ops[50] = NPUOP(OP_REG_CORE, value, CORE_DATAOUT_SIZE_0);
  value = core_desc->dataout_channel & 0xFFFF;
  ops[51] = NPUOP(OP_REG_CORE, value, CORE_DATAOUT_SIZE_1);
  ops[52] = NPUOP(OP_REG_CORE, 0x0, CORE_CLIP_TRUNCATE);
  ops[53] = NPUOP(OP_REG_CORE, 0x0, CORE_3030);
  value = ((dpu_desc->burst_len & 0xF) << 5) | ((dpu_desc->conv_mode & 0x3) <<3) |
    ((dpu_desc->output_mode & 0x3) <<1) | (dpu_desc->flying_mode & 0x1);
  ops[54] = NPUOP(OP_REG_DPU, value, DPU_FEATURE_MODE_CFG);
  value = ((dpu_desc->out_precision & 0x7) << 29) | ((dpu_desc->in_precision & 0x7) << 26) |
    (dpu_desc->proc_precision & 0x7);
  ops[55] = NPUOP(OP_REG_DPU, value, DPU_DATA_FORMAT);
  ops[56] = NPUOP(OP_REG_DPU, 0x0, DPU_OFFSET_PEND);
  ops[57] = NPUOP(OP_REG_DPU, dpu_desc->dst_base_addr, DPU_DST_BASE_ADD);
  value = (dpu_desc->dst_surf_stride & 0xFFFFFFF) << 4;
  ops[58] = NPUOP(OP_REG_DPU, value, DPU_DST_SURF_STRIDE);
  value = dpu_desc->width & 0x1FFF;
  ops[59] = NPUOP(OP_REG_DPU, value, DPU_DATA_CUBE_WIDTH);
  value = dpu_desc->height & 0x1FFF;
  ops[60] = NPUOP(OP_REG_DPU, value, DPU_DATA_CUBE_HEIGHT);
  ops[61] = NPUOP(OP_REG_DPU, 0x0, DPU_DATA_CUBE_NOTCH_ADDR);
  value = ((dpu_desc->channel & 0x1FFF) << 16) | (dpu_desc->channel & 0x1FFF);
  ops[62] = NPUOP(OP_REG_DPU, value, DPU_DATA_CUBE_CHANNEL);
  value = ((dpu_desc->bs_relu_bypass & 0x1) << 6) | ((dpu_desc->bs_mul_bypass & 0x1) << 4) | 
    ((dpu_desc->bs_alu_bypass & 0x1) << 1) | (dpu_desc->bs_bypass & 0x1);
  ops[63] = NPUOP(OP_REG_DPU, value, DPU_BS_CFG);
  ops[64] = NPUOP(OP_REG_DPU, 0x0, DPU_BS_ALU_CFG);
  ops[65] = NPUOP(OP_REG_DPU, 0x0, DPU_BS_MUL_CFG);
  ops[66] = NPUOP(OP_REG_DPU, 0x0, DPU_BS_RELUX_CMP_VALUE);
  value = ((dpu_desc->size_e_2 & 0x7) << 8) | ((dpu_desc->size_e_1 & 0x7) << 5) | 
    ((dpu_desc->size_e_0 & 0x7) << 2) | ((dpu_desc->od_bypass & 0x1) << 1);
  ops[67] = NPUOP(OP_REG_DPU, value,  DPU_BS_OW_CFG);
  ops[68] = NPUOP(OP_REG_DPU, 0x0, DPU_BS_OW_OP);
  value = dpu_desc->channel_wdma & 0x1FFF;
  ops[69] = NPUOP(OP_REG_DPU, value, DPU_WDMA_SIZE_0);
  value = ((dpu_desc->height_wdma & 0x1FFF) << 16) | (dpu_desc->width_wdma & 0x1FFF);
  ops[70] = NPUOP(OP_REG_DPU, value, DPU_WDMA_SIZE_1);
  value = ((dpu_desc->  bn_relu_bypass & 0x1) << 6) |((dpu_desc->bn_mul_bypass &0x1) << 4) | 
    ((dpu_desc->bn_alu_bypass & 0x1) << 1) | (dpu_desc->bn_bypass & 0x1);
  ops[71] = NPUOP(OP_REG_DPU, value, DPU_BN_CFG);
  ops[72] = NPUOP(OP_REG_DPU, 0x0, DPU_BN_ALU_CFG);
  ops[73] = NPUOP(OP_REG_DPU, 0x0, DPU_BN_MUL_CFG);
  ops[74] = NPUOP(OP_REG_DPU, 0x0,DPU_BN_RELUX_CMP_VALUE);
  value = ((dpu_desc->ew_relu_bypass & 0x1) << 9) | ((dpu_desc->ew_op_cvt_bypass & 0x1) << 8) | 
    ((dpu_desc->ew_lut_bypass & 0x1) <<7) | ((dpu_desc->ew_op_bypass & 0x1) << 1) | 
    (dpu_desc->ew_bypass & 0x1);
  ops[75] = NPUOP(OP_REG_DPU, value, DPU_EW_CFG);
  ops[76] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_CVT_OFFSET_VALUE);
  ops[77] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_CVT_SCALE_VALUE);
  ops[78] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_RELUX_CMP_VALUE);
  ops[79] = NPUOP(OP_REG_DPU, 0x0, DPU_OUT_CVT_OFFSET);
  ops[80] = NPUOP(OP_REG_DPU, 0x0, DPU_OUT_CVT_SCALE);
  ops[81] = NPUOP(OP_REG_DPU, 0x0, DPU_OUT_CVT_SHIFT);
  ops[82] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_0);
  ops[83] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_1);
  ops[84] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_2);
  ops[85] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_3);
  ops[86] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_4);
  ops[87] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_5);
  ops[88] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_6);
  ops[89] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_7);
  value = ((dpu_desc->surf_add & 0xFFFFFFF) << 4);
  ops[90] = NPUOP(OP_REG_DPU, 0x0, DPU_SURFACE_ADD);
  ops[91] = NPUOP(OP_REG_DPU, 0x0, DPU_40C4);
  ops[92] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_ACCESS_CFG);
  ops[93] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_ACCESS_DATA);
  ops[94] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_CFG);
  ops[95] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_INFO);
  ops[96] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LE_START);
  ops[97] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LE_END);
  ops[98] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LO_START);
  ops[99] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LO_END);
  ops[100] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LE_SLOPE_SCALE);
  ops[101] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LE_SLOPE_SHIFT);
  ops[102] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LO_SLOPE_SCALE);
  ops[103] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LO_SLOPE_SHIFT);
  ops[104] = NPUOP(OP_REG_PC, 0x0, 0x0);
  ops[105] = NPUOP(OP_REG_PC, 0x0, PC_REGISTER_AMOUNTS);
  ops[106] = NPUOP(OP_40, 0x0, 0x0);
  ops[107] = NPUOP(OP_ENABLE, (PC_ENABLE_DPU | PC_ENABLE_CNA | PC_ENABLE), PC_OPERATION_ENABLE);
 
  return ret;
}

/* 
 * Simpify version of matrix mutliplication because :
 * a) we fail if cbuf storage is exceeded ie M,K,N get too large
 * b) because of (a) only generates a single task
 *
 * task memory needs to hold at laest 112 values
 * TODO: Fix a) & b) 
 * 
 * */
int gen_matmul_fp16(uint16_t M, uint16_t K, uint16_t N, uint32_t input, uint32_t weights, uint32_t output, uint64_t *task) {

   npu_cna_desc cna_desc;
   npu_core_desc core_desc;
   npu_dpu_desc dpu_desc;

   unsigned int fd_bytes;
   unsigned int fd_banks;
   unsigned int weight_banks;
   int surf_stride;

   cna_desc.conv_mode = direct_convolution;
   cna_desc.in_precision = precision_float16;
   cna_desc.proc_precision = precision_float16;

   cna_desc.feature_grains = 1;
   cna_desc.conv_x_stride = 1;
   cna_desc.conv_y_stride = 1;

   cna_desc.datain_width = 1;
   cna_desc.datain_height = M;
   cna_desc.datain_channel = K;
   cna_desc.dataout_width = 1;
   cna_desc.dataout_height = M;
   cna_desc.dataout_atomics = cna_desc.dataout_width * cna_desc.dataout_height;

   cna_desc.weight_width = 1;
   cna_desc.weight_height = 1;
   cna_desc.weight_kernels = N;
   cna_desc.weight_bytes_per_kernel = cna_desc.weight_width * cna_desc.weight_height * 
     cna_desc.datain_channel * sizeof(__fp16);
   cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels; 

   fd_bytes = cna_desc.datain_width * cna_desc.datain_height * cna_desc.datain_channel * sizeof(__fp16);
   fd_banks = (fd_bytes / NPU_CBUF_BANK_SIZE);
   fd_banks = ((fd_bytes % NPU_CBUF_BANK_SIZE) == 0) ? fd_banks : fd_banks +1;
   weight_banks = (cna_desc.weight_bytes / NPU_CBUF_BANK_SIZE);
   weight_banks = ((cna_desc.weight_bytes % NPU_CBUF_BANK_SIZE)==0) ? weight_banks : weight_banks + 1;
   if ((fd_banks + weight_banks) > NPU_CBUF_BANKS) {
     return -1;
   } else {
     if (fd_banks + weight_banks < NPU_CBUF_BANKS) {
       weight_banks = NPU_CBUF_BANKS - fd_banks;
     }
   }
   cna_desc.weight_bank = weight_banks;
   cna_desc.data_bank = fd_banks;
   cna_desc.data_entries = (cna_desc.datain_width * cna_desc.datain_channel) / 32;
   cna_desc.data_entries = (((cna_desc.datain_width * cna_desc.datain_channel) % 32) == 0) ? 
     cna_desc.data_entries : cna_desc.data_entries +1;
   cna_desc.data_sign = 0x0;
   cna_desc.cvt_bypass = 0x1;
   cna_desc.cvt_scale0 = 0x1;
   cna_desc.cvt_scale1 = 0x1;
   cna_desc.cvt_scale2 = 0x1;
   cna_desc.cvt_scale3 = 0x1;
   cna_desc.fc_skip_en = 0;
   cna_desc.data_offset = 0x0;
   cna_desc.pad_left = 0;
   cna_desc.pad_top = 0;
   cna_desc.feature_base_addr = input;
   cna_desc.weight_offset = 0;
   cna_desc.weight_burst_len = 0xf;
   cna_desc.data_burst_len = 0xf;
   cna_desc.line_stride = cna_desc.datain_width * 4;
   surf_stride = cna_desc.line_stride * ((cna_desc.datain_height / 4)-1);
   surf_stride = surf_stride < 0 ? surf_stride + 1 : surf_stride;
   cna_desc.surf_stride = surf_stride;
   cna_desc.dma_width = cna_desc.datain_width;
   cna_desc.dma_height = cna_desc.datain_height;
   cna_desc.dma_channel = cna_desc.datain_channel;
   cna_desc.decompress_addr0 = weights;

   core_desc.proc_precision = precision_float16;
   core_desc.qd_en = 1;
   core_desc.dataout_height = cna_desc.dataout_height - 1;
   core_desc.dataout_width = cna_desc.dataout_width - 1;
   core_desc.dataout_channel = cna_desc.weight_kernels -1;

   dpu_desc.burst_len = 0xf;
   dpu_desc.conv_mode = direct_convolution;
   dpu_desc.output_mode = 0x2;
   dpu_desc.flying_mode = 0x0;
   dpu_desc.out_precision = precision_float32;
   dpu_desc.in_precision = precision_float16;
   dpu_desc.proc_precision = precision_float32;
   dpu_desc.dst_base_addr = output;
   dpu_desc.dst_surf_stride = cna_desc.dataout_height * cna_desc.dataout_width;
   dpu_desc.width = core_desc.dataout_width ;
   dpu_desc.height = core_desc.dataout_height;
   dpu_desc.channel = core_desc.dataout_channel;
   dpu_desc.bs_bypass = 1;
   dpu_desc.bs_alu_bypass = 1;
   dpu_desc.bs_mul_bypass = 1;
   dpu_desc.bs_relu_bypass = 1;
   dpu_desc.size_e_2 = 3; 
   dpu_desc.size_e_1 = 3; 
   dpu_desc.size_e_0 = 3; 
   dpu_desc.width_wdma = core_desc.dataout_width;
   dpu_desc.height_wdma = core_desc.dataout_height;
   dpu_desc.channel_wdma = core_desc.dataout_channel;

   gen_matmul_task(task,&cna_desc,&core_desc,&dpu_desc);

   return 0;
}
