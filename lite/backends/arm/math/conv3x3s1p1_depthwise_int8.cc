// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <arm_neon.h>
#include "lite/backends/arm/math/conv_depthwise.h"
#include "lite/backends/arm/math/type_trans.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
template <typename Dtype>
void conv_depthwise_3x3s1p1_bias_int8(Dtype* dout,
                                      const int8_t* din,
                                      const int8_t* weights,
                                      const float* scale,
                                      const float* bias,
                                      bool flag_bias,
                                      int num,
                                      int chin,
                                      int hin,
                                      int win,
                                      int hout,
                                      int wout,
                                      ARMContext* ctx);

template <typename Dtype>
void conv_depthwise_3x3s1p1_bias_s_int8(Dtype* dout,
                                      const int8_t* din,
                                      const int8_t* weights,
                                      const float* scale,
                                      const float* bias,
                                      bool flag_bias,
                                      int num,
                                      int chin,
                                      int hin,
                                      int win,
                                      int hout,
                                      int wout,
                                      ARMContext* ctx);

template <typename Dtype>
void conv_depthwise_3x3s1p1_bias_relu_int8(Dtype* dout,
                                      const int8_t* din,
                                      const int8_t* weights,
                                      const float* scale,
                                      const float* bias,
                                      bool flag_bias,
                                      int num,
                                      int chin,
                                      int hin,
                                      int win,
                                      int hout,
                                      int wout,
                                      ARMContext* ctx);

template <typename Dtype>
void conv_depthwise_3x3s1p1_bias_s_relu_int8(Dtype* dout,
                                      const int8_t* din,
                                      const int8_t* weights,
                                      const float* scale,
                                      const float* bias,
                                      bool flag_bias,
                                      int num,
                                      int chin,
                                      int hin,
                                      int win,
                                      int hout,
                                      int wout,
                                      ARMContext* ctx);

template <typename Dtype>
void conv_depthwise_3x3s1p1_int8(Dtype* dout,
                                 const int8_t* din,
                                 const int8_t* weights,
                                 const float* scale,
                                 const float* bias,
                                 bool flag_bias,
                                 bool flag_act,
                                 int num,
                                 int chin,
                                 int hin,
                                 int win,
                                 int hout,
                                 int wout,
                                 ARMContext* ctx) {
  if (flag_act) {
    if (win > 8) {
      conv_depthwise_3x3s1p1_bias_relu_int8(dout, din, weights, scale, bias, flag_bias, num, chin, hin, win, hout, wout, ctx);
    } else {
      conv_depthwise_3x3s1p1_bias_s_relu_int8(dout, din, weights, scale, bias, flag_bias, num, chin, hin, win, hout, wout, ctx);
    }
  } else {
    if (win > 8) {
      conv_depthwise_3x3s1p1_bias_int8(dout, din, weights, scale, bias, flag_bias, num, chin, hin, win, hout, wout, ctx);
    } else {
      conv_depthwise_3x3s1p1_bias_s_int8(dout, din, weights, scale, bias, flag_bias, num, chin, hin, win, hout, wout, ctx);
    }
  }
}
/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias, width > 4
 */
template <typename Dtype>
void conv_depthwise_3x3s1p1_bias_int8(Dtype* dout,
                                      const int8_t* din,
                                      const int8_t* weights,
                                      const float* scale,
                                      const float* bias,
                                      bool flag_bias,
                                      int num,
                                      int ch_in,
                                      int h_in,
                                      int w_in,
                                      int h_out,
                                      int w_out,
                                      ARMContext* ctx) {
    //! pad is done implicit
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint8_t right_pad_idx[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    int8_t* zero_ptr = ctx->workspace_data<int8_t>();
    memset(zero_ptr, 0, w_in * sizeof(int8_t));
    int* write_ptr = reinterpret_cast<int*>(zero_ptr) + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = (w_in + 7) >> 3;
    int tile_h = (h_out + 1) >> 1;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(w_in - 7 - (cnt_col << 3));

    int size_pad_bottom = h_out % 2;

    uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));

    uint8x16_t vmask_rp = vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
    // uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    uint8_t vmask[16];
    vst1q_u8(vmask, vmask_rp);

    unsigned int rst_remain = (unsigned int)(w_out - ((cnt_col + 1) << 3));
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);

    int8x8_t vzero = vdup_n_s8(0);
    int32x4_t vzero_32 = vdupq_n_s32(0);

    int32_t* pre_out = reinterpret_cast<int*>(write_ptr + w_out + 4);

    for (int n = 0; n < num; ++n) {
        const int8_t *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = pre_out + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const int8_t* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? static_cast<int>(bias[c] / scale[c]) : 0;

            const int8_t* wei_ptr = weights + c * w_stride;

#ifdef __aarch64__
            int vbias[4] = {bias_val, bias_val, bias_val, bias_val};

            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);
#endif
            int *doutr0 = nullptr;
            int *doutr1 = nullptr;

            const int8_t *dr0 = din_ch_ptr;
            const int8_t *dr1 = dr0 + w_in;
            const int8_t *dr2 = dr1 + w_in;
            const int8_t *dr3 = dr2 + w_in;

            const int8_t *din_ptr0 = nullptr;
            const int8_t *din_ptr1 = nullptr;
            const int8_t *din_ptr2 = nullptr;
            const int8_t *din_ptr3 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                unsigned int* rst_mask = rmask;
                uint8_t* val_mask = vmask;

                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    din_ptr3 = dr2;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr3;
                    dr3 = dr2 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr3;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                }
                //! process bottom pad
                if (i + 3 > h_in) {
                    switch (i + 3 - h_in) {
                        case 3:
                            din_ptr1 = zero_ptr;
                        case 2:
                            din_ptr2 = zero_ptr;
                        case 1:
                            din_ptr3 = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 2 > h_out) {
                    doutr1 = write_ptr;
                }
                int cnt = cnt_col;
#ifdef __aarch64__
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr3]] \n"
                    "movi   v21.4s, #0x0\n"                         /* out0 = 0 */
                // left
                    "ld1    {v0.8b}, [%[din_ptr0]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v1.8b}, [%[din_ptr0]]                   \n"        /* load a00-a015 to q0*/
                    "ld1    {v3.8b}, [%[din_ptr1]]                   \n"        /* load a00-a015 to q0*/

                    "ld1    {v10.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v11.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v12.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v13.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    //r0
                    "smull  v18.8h,  %[v1].8b,  v0.8b   \n"       /* outr00 = 01234567 * w01 */

                    "ext v4.8b, v21.8b, v0.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v5.8b, v0.8b, v1.8B, #1       \n" /* vext_s8(vinr0, vinr0_1, 1); 12345678 */

                    "ld1    {v6.8b}, [%[din_ptr2]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v8.8b}, [%[din_ptr3]], #8                       \n"        /* load a00-a015 to q0*/

                    "smlal  v18.8h,  %[v0].8b,  v4.8b\n"         /* outr00 += 00123456 * w00 */

                    "ld1    {v7.8b}, [%[din_ptr2]]                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v9.8b}, [%[din_ptr3]]                       \n"        /* load a00-a015 to q0*/

                    "sub   %[din_ptr0], %[din_ptr0], #1                       \n"
                    "sub   %[din_ptr1], %[din_ptr1], #1                       \n"

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v2].8b,  v5.8b\n"         /* outr00 += 12345678 * w02 */

                    "ext v4.8b, v21.8b, v2.8b, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v5.8b, v2.8b, v3.8b, #1       \n" /* vext_s8(vinr0, vinr0_1, 1); 12345678 */

                    //r1
                    "sub   %[din_ptr2], %[din_ptr2], #1                       \n"
                    "sub   %[din_ptr3], %[din_ptr3], #1                       \n"

                    "smull  v19.8h,  %[v1].8b,  v2.8b   \n"       /* outr10 += 01234567 * w11 */
                    "smlal  v18.8h,  %[v4].8b,  v2.8b   \n"       /* outr00 += 01234567 * w11 */

                    "ext v14.8b, v21.8b, v6.8b, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v15.8b, v6.8b, v7.8b, #1       \n" /* vext_s8(vinr0, vinr0_1, 1); 12345678 */

                    "smlal  v19.8h,  %[v0].8b,  v4.8b   \n"       /* outr00 += 01234567 * w11 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v3].8b,  v4.8b   \n"       /* outr00 += 001234567 * w10 */

                    "ld1    {v0.8b}, [%[din_ptr0]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8                       \n"        /* load a00-a015 to q0*/

                    "saddw   v12.4s, v12.4s, v19.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v2].8b,  v5.8b   \n"       /* outr00 += 01234567 * w11 */
                    "smlal  v18.8h,  %[v5].8b,  v5.8b   \n"       /* outr00 += 12345678 * w12 */

                    //r2
                    "ld1    {v1.8b}, [%[din_ptr0]]                   \n"        /* load a00-a015 to q0*/
                    "ld1    {v3.8b}, [%[din_ptr1]]                   \n"        /* load a00-a015 to q0*/

                    "smlal  v19.8h,  %[v4].8b,  v6.8b   \n"       /* outr10 += 01234567 * w11 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v7].8b,  v6.8b   \n"       /* outr00 += 01234567 * w11 */

                    "ext v4.8b, v21.8b, v8.8b, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v5.8b, v8.8b, v9.8b, #1       \n" /* vext_s8(vinr0, vinr0_1, 1); 12345678 */

                    "saddw   v12.4s, v12.4s, v19.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v3].8b,  v14.8b   \n"       /* outr10 += 01234567 * w11 */
                    "smlal  v18.8h,  %[v6].8b,  v14.8b   \n"       /* outr00 += 01234567 * w11 */

                    "ld1    {v6.8b}, [%[din_ptr2]], #8                       \n"        /* load a00-a015 to q0*/

                    "smlal  v19.8h,  %[v5].8b,  v15.8b   \n"       /* outr10 += 01234567 * w11 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v8].8b,  v15.8b   \n"       /* outr00 += 01234567 * w11 */

                    "saddw   v12.4s, v12.4s, v19.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"             /* v11 += outr00.high*/

                    //r3
                    "smull  v19.8h,  %[v7].8b,  v8.8b   \n"       /* outr00 += 01234567 * w11 */

                    "ld1    {v8.8b}, [%[din_ptr3]], #8                       \n"        /* load a00-a015 to q0*/

                    "ld1    {v7.8b}, [%[din_ptr2]]                   \n"        /* load a00-a015 to q0*/
                    "ld1    {v9.8b}, [%[din_ptr3]]                   \n"        /* load a00-a015 to q0*/

                    "smlal  v19.8h,  %[v6].8b,  v4.8b     \n"       /* outr00 += 01234567 * w11 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "stp     q10, q11, [%[ptr_out0]], #32 \n"      /* store q10, q11 -> ptr_out       */

                    "saddw   v12.4s, v12.4s, v19.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v8].8b,  v5.8b        \n"       /* outr00 += 01234567 * w11 */

                    "ld1    {v10.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v11.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    "saddw   v12.4s, v12.4s, v19.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"             /* v11 += outr00.high*/

                    "stp     q12, q13, [%[ptr_out1]], #32   \n"      /* store q10, q11 -> ptr_out       */

                    "ld1    {v12.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v13.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    "cmp  %[cnt], #1                \n"
                    "blt 3f                         \n"
                //mid
                    "1:                             \n"
                    "ext v4.8b, v0.8B, v1.8b, #1       \n" /*12345678 */
                    "ext v5.8b, v0.8b, v1.8B, #2       \n" /*23456789 */

                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ext v14.8b, v2.8B, v3.8b, #1       \n" /*12345678 */
                    "ext v15.8b, v2.8b, v3.8B, #2       \n" /*23456789 */

                    "smlal  v18.8h,  %[v1].8b,  v4.8b\n"         /* outr00 += 12345678 * w01 */

                    "ext v16.8b, v6.8B, v7.8b, #1       \n" /*12345678 */
                    "ext v17.8b, v6.8b, v7.8B, #2       \n" /*23456789 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v2].8b,  v5.8b\n"         /* outr00 += 23456789 * w02 */

                    //r1
                    "ext v4.8b, v8.8B, v9.8b, #1       \n" /*12345678 */
                    "ext v5.8b, v8.8b, v9.8B, #2       \n" /*23456789 */

                    "smull  v19.8h,  %[v0].8b,  v2.8b   \n"       /* outr00 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ld1    {v0.8b}, [%[din_ptr0]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8                       \n"        /* load a00-a015 to q0*/

                    "smlal  v19.8h,  %[v1].8b,  v14.8b\n"         /* outr00 += 12345678 * w01 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v4].8b,  v14.8b\n"         /* outr00 += 12345678 * w01 */

                    "ld1    {v1.8b}, [%[din_ptr0]]                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v3.8b}, [%[din_ptr1]]                       \n"        /* load a00-a015 to q0*/

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v2].8b,  v15.8b\n"         /* outr00 += 23456789 * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b\n"         /* outr00 += 12345678 * w01 */

                    //r2
                    "smlal  v19.8h,  %[v3].8b,  v6.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v6].8b,  v6.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v4].8b,  v16.8b\n"         /* outr00 += 12345678 * w01 */
                    "smlal  v18.8h,  %[v7].8b,  v16.8b\n"         /* outr00 += 12345678 * w01 */

                    "smlal  v19.8h,  %[v5].8b,  v17.8b\n"         /* outr00 += 23456789 * w02 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v8].8b,  v17.8b\n"         /* outr00 += 12345678 * w01 */

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ld1    {v6.8b}, [%[din_ptr2]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v8.8b}, [%[din_ptr3]], #8                       \n"        /* load a00-a015 to q0*/

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smlal  v19.8h,  %[v7].8b,  v4.8b\n"         /* outr00 += 12345678 * w01 */

                    "ld1    {v7.8b}, [%[din_ptr2]]                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v9.8b}, [%[din_ptr3]]                       \n"        /* load a00-a015 to q0*/

                    "stp     q10, q11, [%[ptr_out0]], #32 \n"      /* store q10, q11 -> ptr_out       */

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v8].8b,  v5.8b\n"         /* outr00 += 23456789 * w02 */

                    "ld1    {v10.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v11.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "subs %[cnt], %[cnt], #1            \n"

                    "stp     q12, q13, [%[ptr_out1]], #32 \n"      /* store q10, q11 -> ptr_out       */

                    "ld1    {v12.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v13.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    "bne 1b                                 \n"
                //right
                    "3:                             \n"
                    "ld1 {v14.8b}, [%[vmask]], #8             \n"
                    "ld1 {v15.8b}, [%[vmask]]                \n"

                    "bif v0.8b, v21.8b, v14.8b               \n"
                    "bif v1.8b, v21.8b, v15.8b               \n"
                    "bif v2.8b, v21.8b, v14.8b               \n"
                    "bif v3.8b, v21.8b, v15.8b               \n"

                    "ext v4.8b, v0.8b, v1.8b, #1             \n"
                    "ext v5.8b, v0.8b, v1.8b, #2             \n"

                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ext v16.8b, v2.8b, v3.8b, #1             \n"
                    "ext v17.8b, v2.8b, v3.8b, #2             \n"

                    "bif v6.8b, v21.8b, v14.8b               \n"
                    "bif v7.8b, v21.8b, v15.8b               \n"

                    "smlal  v18.8h,  %[v1].8b,  v4.8b   \n"       /* outr00 = 01234567 * w00 */

                    "bif v8.8b, v21.8b, v14.8b               \n"
                    "bif v9.8b, v21.8b, v15.8b               \n"

                    "ext v20.8b, v6.8b, v7.8b, #1             \n"
                    "ext v22.8b, v6.8b, v7.8b, #2             \n"

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v2].8b,  v5.8b   \n"       /* outr00 = 01234567 * w00 */

                    //r1
                    "ext v4.8b, v8.8b, v9.8b, #1             \n"
                    "ext v5.8b, v8.8b, v9.8b, #2             \n"

                    "smull  v19.8h,  %[v0].8b,  v2.8b   \n"       /* outr00 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ld1 {v14.4s}, [%[rmask]], #16                \n"
                    "ld1 {v15.4s}, [%[rmask]]                     \n"

                    "smlal  v19.8h,  %[v1].8b,  v16.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v4].8b,  v16.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ld1 {v0.4s}, [%[ptr_out0]], #16                \n"
                    "ld1 {v2.4s}, [%[ptr_out1]], #16                \n"

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v2].8b,  v17.8b   \n"       /* outr00 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v5].8b,  v17.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ld1 {v1.4s}, [%[ptr_out0]]                   \n"
                    "ld1 {v3.4s}, [%[ptr_out1]]                   \n"

                    //r2
                    "smlal  v19.8h,  %[v3].8b,  v6.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v6].8b,  v6.8b   \n"       /* outr00 = 01234567 * w00 */

                    "sub %[ptr_out0], %[ptr_out0], #16   \n"
                    "sub %[ptr_out1], %[ptr_out1], #16   \n"

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v4].8b,  v20.8b   \n"       /* outr00 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v7].8b,  v20.8b   \n"       /* outr00 = 01234567 * w00 */

                    "smlal  v19.8h,  %[v5].8b,  v22.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v8].8b,  v22.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smlal  v19.8h,  %[v7].8b,  v4.8b   \n"       /* outr00 = 01234567 * w00 */

                    "bif v10.16b, v0.16b, v14.16b         \n"
                    "bif v11.16b, v1.16b, v15.16b         \n"

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v8].8b,  v5.8b   \n"       /* outr00 = 01234567 * w00 */

                    "stp     q10, q11, [%[ptr_out0]], #32 \n"      /* store q10, q11 -> ptr_out       */

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "bif v12.16b, v2.16b, v14.16b         \n"
                    "bif v13.16b, v3.16b, v15.16b         \n"

                    "stp     q12, q13, [%[ptr_out1]], #32 \n"      /* store q10, q11 -> ptr_out       */

                    :[cnt]"+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [din_ptr3] "+r"(din_ptr3), [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
                        [vmask] "+r" (val_mask), [rmask] "+r" (rst_mask)
                    : [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), [bias_val] "r" (vbias), \
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22)
                    :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12","v13","v14","v15", "v16", "v17", "v18",\
                      "v19", "v20", "v21", "v22"
                );
#else
                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vmov.u32 d11, #0                   @ zero\n"
                    //out0
                    "vdup.32 q8, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                            @ and \n" //q9 = vbias
                    //out1
                    "vdup.32 q10, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                            @ and \n" //q9 = vbias

                    //r0
                    "vmull.s8 q12, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678

                    "vld1.8 {d12-d13}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vdup.s8     d5, d0[3]               @ d5 = w10, w10, w00, w00\n"
                    "vdup.s8     d6, d0[4]               @ d6 = w11, w11, w01, w01\n"

                    "vmlal.s8 q12, d30, d2                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                    "vdup.s8     d7, d0[5]               @ d7 = w12, w12\n"
                    "add %[din_ptr0], #7                   @add \n"
                    "add %[din_ptr1], #7                   @add \n"

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d3                 @ out1 = din1 * w01 \n" // q13 = d12 * w01

                    "vmlal.s8 q12, d12, d6                 @ out0 = din1 * w11 \n" // q12 = d12 * w11

                    "vld1.8 {d12-d13}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vdup.s8     d8, d0[6]               @ d8 = w20, w00, w00, w00\n"
                    "vdup.s8     d9, d0[7]               @ d9 = w21, w01, w01, w01\n"
                    "vdup.s8     d10, d1[0]               @ d10 = w22, w02, w02, w02\n"

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d30, d2                 @ out1 += din1 * w00 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d5                 @ out0 += din1 * w10 \n" // q12 += d10 * w00

                    "add %[din_ptr2], #7                   @add \n"
                    "add %[din_ptr3], #7                   @add \n"

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #1          @ ext \n" //d11 = 12345678

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d14, d6                 @ out1 = din2 * w11 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d9                 @ out1 = din2 * w21 \n" // q13 = d12 * w01

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d30, d5                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d8                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d12, d9                 @ out1 = din3 * w21 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"

                    "vmlal.s8 q13, d30, d8                 @ out1 += din3 * w20 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vst1.32 {d16-d17}, [%[dout_ptr1]]!         @ store\n"

                    "vmull.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vst1.32 {d20-d21}, [%[dout_ptr2]]!         @ store\n"
                    "cmp %[cnt], #1                                 \n"
                    "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"
                    "blt 1f                                         \n"

                //mid
                    "2:                                          \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    //out0
                    "vdup.32 q8, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                            @ and \n" //q9 = vbias
                    //out1
                    "vdup.32 q10, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                            @ and \n" //q9 = vbias

                     //r0
                    "vmull.s8 q12, d12, d2                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vext.8     d30, d12, d13, #1     @ ext \n" //d10 = 12345678
                    "vext.8     d31, d12, d13, #2          @ ext \n" //d11 = 23456789

                    "vld1.8 {d12-d13}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q12, d30, d3                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                    "add %[din_ptr0], #8                   @add \n"
                    "add %[din_ptr1], #8                   @add \n"
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8     d30, d12, d13, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          @ ext \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d2                 @ out1 = din1 * w01 \n" // q13 = d12 * w01

                    "vmlal.s8 q12, d12, d5                 @ out0 = din1 * w11 \n" // q12 = d12 * w11

                    "vld1.8 {d12-d13}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d30, d3                 @ out1 += din1 * w00 \n" // q12 += d10 * w00

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q12, d30, d6                 @ out0 += din1 * w10 \n" // q12 += d10 * w00

                    "add %[din_ptr2], #8                   @add \n"
                    "add %[din_ptr3], #8                   @add \n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 12345678

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d14, d5                 @ out1 = din2 * w11 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d8                 @ out1 = din2 * w21 \n" // q13 = d12 * w01

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d30, d6                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d12, d13, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d12, d8                 @ out1 = din3 * w21 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"

                    "vmlal.s8 q13, d30, d9                 @ out1 += din3 * w20 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"

                    "vst1.32 {d16-d17}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vst1.32 {d20-d21}, [%[dout_ptr2]]!         @ store\n"
                    "subs %[cnt], #1                                \n"
                    "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"
                    "bne  2b                                        \n"
                //right
                    "1:                                          \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    //out0
                    "vdup.32 q8, %[bias]                 @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                 @ and \n" //q9 = vbias
                    //out1
                    "vdup.32 q10, %[bias]                @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                @ and \n" //q9 = vbias

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"
                    "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                     //r0
                    "vmull.s8 q12, d12, d2                 @ out0 = din0 * w00 \n" // q12 = d12 * w01
                    "vext.8 d30, d12, d13, #1               @ ext \n" //d10 = 12345678
                    "vext.8 d31, d12, d13, #2               @ ext \n" //d11 = 23456789

                    "vld1.8 {d12-d13}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                    "vmlal.s8 q12, d30, d3                 @ out0 += din0 * w01 \n" // q12 += d10 * w00

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8 d30, d14, d15, #1           @ ext \n" //d10 = 00123456
                    "vext.8 d31, d14, d15, #2          @ ext \n" //d11 = 12345678

                    "vmull.s8 q13, d14, d2                 @ out1 = din1 * w00 \n" // q13 = d12 * w01

                    "vmlal.s8 q12, d14, d5                 @ out0 = din1 * w10 \n" // q12 = d12 * w11

                    "vld1.8 {d14-d15}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vbif.8 d12, d11, d28                 @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29                 @ bit select, deal with right pad\n"

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d30, d3                 @ out1 += din1 * w01 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d6                 @ out0 += din1 * w11 \n" // q12 += d10 * w00


                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8 d30, d12, d13, #1               @ ext \n" //d10 = 00123456
                    "vext.8 d31, d12, d13, #2               @ ext \n" //d11 = 12345678

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d12, d5                 @ out1 = din2 * w10 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d12, d8                 @ out1 = din2 * w20 \n" // q13 = d12 * w01

                    "vbif.8 d14, d11, d28                     @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29                     @ bit select, deal with right pad\n"

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d30, d6                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    "vld1.32 {d28-d29}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d12-d13}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d14, d8                 @ out1 = din3 * w20 \n" // q13 = d12 * w01
                    "sub %[dout_ptr1], #16                  @ sub \n"
                    "vld1.32 {d14-d15}, [%[dout_ptr2]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d24-d25}, [%[dout_ptr2]]     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d30, d9                 @ out1 += din3 * w21 \n" // q13 += d10 * w00
                    "vbif q8, q14, q1                   @ bit select, deal with right pad\n"
                    "vbif q9, q6, q2                    @ bit select, deal with right pad\n"
                    "sub %[dout_ptr2], #16                  @ sub \n"

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    "vst1.32 {d16-d17}, [%[dout_ptr1]]!         @ store\n"
                    "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vbif q10, q7, q1        @ bit select, deal with right pad\n"
                    "vbif q11, q12, q2       @ bit select, deal with right pad\n"

                    "vst1.32 {d20-d21}, [%[dout_ptr2]]!         @ store\n"
                    "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [din_ptr3] "+r" (din_ptr3), \
                      [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                      "q11", "q12", "q13", "q14", "q15"
                );
#endif
                dout_ptr += 2 * w_out;
           }
       }
    }
    // write_out
    int32_to_dtype<Dtype>(pre_out, dout, scale, ch_in, num, h_out * w_out);
}

//w_in <= 8
template <typename Dtype>
void conv_depthwise_3x3s1p1_bias_s_int8(Dtype* dout,
                                      const int8_t* din,
                                      const int8_t* weights,
                                      const float* scale,
                                      const float* bias,
                                      bool flag_bias,
                                      int num,
                                      int ch_in,
                                      int h_in,
                                      int w_in,
                                      int h_out,
                                      int w_out,
                                      ARMContext* ctx) {
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const uint8_t right_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    int8_t* zero_ptr = ctx->workspace_data<int8_t>();
    memset(zero_ptr, 0, w_in * sizeof(int8_t));
    int* write_ptr = reinterpret_cast<int*>(zero_ptr) + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_h = (h_out + 1) >> 1;

    unsigned int size_pad_right = (unsigned int)(w_in);

    uint8x8_t vmask_rp = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    // uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    uint8_t vmask[8];
    vst1_u8(vmask, vmask_rp);

    unsigned int rst_remain = (unsigned int)w_out;
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);

    int8x8_t vzero = vdup_n_s8(0);
    int32x4_t vzero_32 = vdupq_n_s32(0);
    int32_t* pre_out = reinterpret_cast<int*>(write_ptr + w_out + 4);

    for (int n = 0; n < num; ++n) {
        const int8_t *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = pre_out + n * ch_in * size_out_channel;
#pragma omp parallel for
         for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const int8_t* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? static_cast<int>(bias[c] / scale[c]) : 0;

            const int8_t* wei_ptr = weights + c * w_stride;
#ifdef __aarch64__
            int vbias[4] = {bias_val, bias_val, bias_val, bias_val};
            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);
#endif
            int *doutr0 = nullptr;
            int *doutr1 = nullptr;

            const int8_t *dr0 = din_ch_ptr;
            const int8_t *dr1 = dr0 + w_in;
            const int8_t *dr2 = dr1 + w_in;
            const int8_t *dr3 = dr2 + w_in;

            const int8_t *din_ptr0 = nullptr;
            const int8_t *din_ptr1 = nullptr;
            const int8_t *din_ptr2 = nullptr;
            const int8_t *din_ptr3 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                unsigned int* rst_mask = rmask;

                int out_buf1[8];
                int out_buf2[8];
                int trash_buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    din_ptr3 = dr2;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr3;
                    dr3 = dr2 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr3;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                }
                //! process bottom pad
                if (i + 3 > h_in) {
                    switch (i + 3 - h_in) {
                        case 3:
                            din_ptr1 = zero_ptr;
                        case 2:
                            din_ptr2 = zero_ptr;
                        case 1:
                            din_ptr3 = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 2 > h_out) {
                    doutr1 = trash_buf;
                }
#ifdef __aarch64__
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr3]] \n"
                    "movi   v21.4s, #0x0\n"                         /* out0 = 0 */
                // left
                    "ld1 {v4.8b}, [%[vmask]]            \n"
                    "ld1    {v0.8b}, [%[din_ptr0]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v1.8b}, [%[din_ptr1]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v2.8b}, [%[din_ptr2]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v3.8b}, [%[din_ptr3]], #8                       \n"        /* load a00-a015 to q0*/

                    "bif v0.8b, v21.8b, v4.8b               \n"
                    "bif v1.8b, v21.8b, v4.8b               \n"
                    "bif v2.8b, v21.8b, v4.8b               \n"
                    "bif v3.8b, v21.8b, v4.8b               \n"

                    "ext v6.8b, v21.8b, v0.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v7.8b, v0.8b, v21.8B, #1       \n" /* vext_s8(vinr0, vinr0_1, 1); 12345678 */

                    "ld1 {v10.4s}, [%[vbias]]            \n"
                    "ld1 {v11.4s}, [%[vbias]]            \n"

                    //r0
                    "smull  v18.8h,  %[v1].8b,  v0.8b   \n"       /* outr00 = 01234567 * w01 */

                    "ext v8.8b, v21.8b, v1.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v9.8b, v1.8b, v21.8B, #1       \n" /* vext_s8(vinr0, vinr0_1, 1); 12345678 */

                    "smlal  v18.8h,  %[v0].8b,  v6.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ld1 {v12.4s}, [%[vbias]]            \n"
                    "ld1 {v13.4s}, [%[vbias]]            \n"

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v2].8b,  v7.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ext v6.8b, v21.8b, v2.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v7.8b, v2.8b, v21.8B, #1       \n" /* vext_s8(vinr0, vinr0_1, 1); 12345678 */

                    //r1
                    "smull  v19.8h,  %[v1].8b,  v1.8b   \n"       /* outr00 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v4].8b,  v1.8b   \n"       /* outr00 = 01234567 * w00 */

                    // "ld1 {v14.4s}, [%[rmask]], #16                \n"
                    // "ld1 {v15.4s}, [%[rmask]]                     \n"

                    "smlal  v19.8h,  %[v0].8b,  v8.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v3].8b,  v8.8b   \n"       /* outr00 = 01234567 * w00 */

                    // "ld1 {v16.4s}, [%[ptr_out0]], #16                \n"
                    // "ld1 {v17.4s}, [%[ptr_out1]], #16                \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v2].8b,  v9.8b   \n"       /* outr00 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v5].8b,  v9.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ext v8.8b, v21.8b, v3.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v9.8b, v3.8b, v21.8B, #1       \n"  //vext_s8(vinr0, vinr0_1, 1); 12345678

                    // "ld1 {v0.4s}, [%[ptr_out0]]                   \n"
                    // "ld1 {v1.4s}, [%[ptr_out1]]                   \n"

                    //r2
                    "smlal  v19.8h,  %[v4].8b,  v2.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v7].8b,  v2.8b   \n"       /* outr00 = 01234567 * w00 */

                    // "sub %[ptr_out0], %[ptr_out0], #16   \n"
                    // "sub %[ptr_out1], %[ptr_out1], #16   \n"

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v3].8b,  v6.8b   \n"       /* outr00 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v6].8b,  v6.8b   \n"       /* outr00 = 01234567 * w00 */

                    "smlal  v19.8h,  %[v5].8b,  v7.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v8].8b,  v7.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    //r3
                    "smull  v19.8h,  %[v7].8b,  v3.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smlal  v19.8h,  %[v6].8b,  v8.8b   \n"       /* outr00 = 01234567 * w00 */

                    // "bif v10.16b, v16.16b, v14.16b         \n"
                    // "bif v11.16b, v0.16b, v15.16b         \n"

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v8].8b,  v9.8b   \n"       /* outr00 = 01234567 * w00 */

                    "stp     q10, q11, [%[ptr_out0]] \n"      /* store q10, q11 -> ptr_out       */

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    // "bif v12.16b, v17.16b, v14.16b         \n"
                    // "bif v13.16b, v1.16b, v15.16b         \n"

                    "stp     q12, q13, [%[ptr_out1]] \n"      /* store q10, q11 -> ptr_out       */

                    :[din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [din_ptr3] "+r"(din_ptr3), \
                        [rmask] "+r" (rst_mask)
                    : [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), [vbias] "r" (vbias), \
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22), [vmask] "r" (vmask), \
                        [ptr_out0] "r"(out_buf1), [ptr_out1] "r"(out_buf2)
                    :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12","v13","v14","v15", "v16", "v17", "v18",\
                      "v19", "v20", "v21", "v22"
                );
#else
                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"
                    "vld1.8 {d28}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d12}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d13}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"

                    "vmov.u32 d11, #0                   @ zero\n"
                    //out0
                    "vdup.32 q8, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                            @ and \n" //q9 = vbias

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d28        @ bit select, deal with right pad\n"
                    "vld1.8 {d14}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d15}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    //out1
                    "vdup.32 q10, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                            @ and \n" //q9 = vbias

                    //r0
                    "vmull.s8 q12, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vext.8 d30, d11, d12, #7           @ ext \n" //d10 = 00123456
                    "vext.8 d31, d12, d11, #1          @ ext \n" //d11 = 12345678

                    "vdup.s8 d5, d0[3]               @ d5 = w10, w10, w00, w00\n"
                    "vdup.s8 d6, d0[4]               @ d6 = w11, w11, w01, w01\n"

                    "vmlal.s8 q12, d30, d2                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                    "vdup.s8 d7, d0[5]               @ d7 = w12, w12\n"
                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d28        @ bit select, deal with right pad\n"

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8     d30, d11, d13, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d13, d11, #1          @ ext \n" //d11 = 12345678
                    "vmull.s8 q13, d13, d3                 @ out1 = din1 * w01 \n" // q13 = d12 * w01

                    "vmlal.s8 q12, d13, d6                 @ out0 = din1 * w11 \n" // q12 = d12 * w11

                    "vdup.s8 d8, d0[6]               @ d8 = w20, w00, w00, w00\n"
                    "vdup.s8 d9, d0[7]               @ d9 = w21, w01, w01, w01\n"
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d30, d2                 @ out1 += din1 * w00 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d5                 @ out0 += din1 * w10 \n" // q12 += d10 * w00

                    "vdup.s8 d10, d1[0]               @ d10 = w22, w02, w02, w02\n"
                    // "vld1.32 {d28-d29}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    // "vld1.32 {d12-d13}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d11, #1          @ ext \n" //d11 = 12345678

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d14, d6                 @ out1 = din2 * w11 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d9                 @ out1 = din2 * w21 \n" // q13 = d12 * w01

                    // "sub %[dout_ptr1], #16                  @ sub \n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d30, d5                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d8                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    // "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    // "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d11, d15, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d15, d11, #1          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d15, d9                 @ out1 = din3 * w21 \n" // q13 = d12 * w01

                    // "vld1.32 {d6-d7}, [%[dout_ptr2]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    // "vld1.32 {d14-d15}, [%[dout_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d30, d8                 @ out1 += din3 * w20 \n" // q13 += d10 * w00

                    // "vbif q8, q14, q1                   @ bit select, deal with right pad\n"
                    // "vbif q9, q6, q2                    @ bit select, deal with right pad\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    // "sub %[dout_ptr2], #16                  @ sub \n"

                    "vst1.32 {d16-d19}, [%[dout_ptr1]]         @ store\n"
                    // "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    // "vbif q10, q3, q1                   @ bit select, deal with right pad\n"
                    // "vbif q11, q7, q2                    @ bit select, deal with right pad\n"

                    "vst1.32 {d20-d23}, [%[dout_ptr2]]         @ store\n"
                    // "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2), \
                      [din_ptr3] "+r" (din_ptr3), \
                      [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask), \
                      [dout_ptr1] "r"(out_buf1), [dout_ptr2] "r"(out_buf2)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                     "q11", "q12", "q13", "q14", "q15"
                );
#endif
                for (int w = 0; w < w_out; ++w){
                    *doutr0++ = out_buf1[w];
                    *doutr1++ = out_buf2[w];
                }
                dout_ptr += 2 * w_out;
            }
        }
    }
    // write_out
    int32_to_dtype<Dtype>(pre_out, dout, scale, ch_in, num, h_out * w_out);
}

template <typename Dtype>
void conv_depthwise_3x3s1p1_bias_relu_int8(Dtype* dout,
                                      const int8_t* din,
                                      const int8_t* weights,
                                      const float* scale,
                                      const float* bias,
                                      bool flag_bias,
                                      int num,
                                      int ch_in,
                                      int h_in,
                                      int w_in,
                                      int h_out,
                                      int w_out,
                                      ARMContext* ctx) {
   // printf("3x3s1 mult height \n");
    //! pad is done implicit
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const uint8_t right_pad_idx[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    int8_t* zero_ptr = ctx->workspace_data<int8_t>();
    memset(zero_ptr, 0, w_in * sizeof(int8_t));
    int* write_ptr = reinterpret_cast<int*>(zero_ptr) + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = (w_in + 7) >> 3;
    int tile_h = (h_out + 1) >> 1;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(w_in - 7 - (cnt_col << 3));

    int size_pad_bottom = h_out % 2;

    uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned int rst_remain = (unsigned int)(w_out - ((cnt_col + 1) << 3));
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    int8x8_t vzero = vdup_n_s8(0);
    int32x4_t vzero_32 = vdupq_n_s32(0);

    uint8x16_t vmask_rp = vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
    // uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    uint8_t vmask[16];
    vst1q_u8(vmask, vmask_rp);

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);
    int32_t* pre_out = reinterpret_cast<int*>(write_ptr + w_out + 4);

    for (int n = 0; n < num; ++n) {
        const int8_t *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = pre_out + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const int8_t* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? static_cast<int>(bias[c] / scale[c]) : 0;

            const int8_t* wei_ptr = weights + c * w_stride;
#ifdef __aarch64__
            int vbias[4] = {bias_val, bias_val, bias_val, bias_val};
            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);
#endif

            int *doutr0 = nullptr;
            int *doutr1 = nullptr;

            const int8_t *dr0 = din_ch_ptr;
            const int8_t *dr1 = dr0 + w_in;
            const int8_t *dr2 = dr1 + w_in;
            const int8_t *dr3 = dr2 + w_in;

            const int8_t *din_ptr0 = nullptr;
            const int8_t *din_ptr1 = nullptr;
            const int8_t *din_ptr2 = nullptr;
            const int8_t *din_ptr3 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                unsigned int* rst_mask = rmask;
                uint8_t* val_mask = vmask;

                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    din_ptr3 = dr2;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr3;
                    dr3 = dr2 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr3;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                }
                //! process bottom pad
                if (i + 3 > h_in) {
                    switch (i + 3 - h_in) {
                        case 3:
                            din_ptr1 = zero_ptr;
                        case 2:
                            din_ptr2 = zero_ptr;
                        case 1:
                            din_ptr3 = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 2 > h_out) {
                    doutr1 = write_ptr;
                }
                int cnt = cnt_col;
#ifdef __aarch64__
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr3]] \n"
                    "movi   v21.4s, #0x0\n"                         /* out0 = 0 */
                // left
                    "ld1    {v0.8b}, [%[din_ptr0]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v1.8b}, [%[din_ptr0]]                   \n"        /* load a00-a015 to q0*/
                    "ld1    {v3.8b}, [%[din_ptr1]]                   \n"        /* load a00-a015 to q0*/

                    "ld1    {v10.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v11.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v12.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v13.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    //r0
                    "smull  v18.8h,  %[v1].8b,  v0.8b   \n"       /* outr00 = 01234567 * w01 */

                    "ext v4.8b, v21.8b, v0.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v5.8b, v0.8b, v1.8B, #1       \n" /* vext_s8(vinr0, vinr0_1, 1); 12345678 */

                    "ld1    {v6.8b}, [%[din_ptr2]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v8.8b}, [%[din_ptr3]], #8                       \n"        /* load a00-a015 to q0*/

                    "smlal  v18.8h,  %[v0].8b,  v4.8b\n"         /* outr00 += 00123456 * w00 */

                    "ld1    {v7.8b}, [%[din_ptr2]]                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v9.8b}, [%[din_ptr3]]                       \n"        /* load a00-a015 to q0*/

                    "sub   %[din_ptr0], %[din_ptr0], #1                       \n"
                    "sub   %[din_ptr1], %[din_ptr1], #1                       \n"

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v2].8b,  v5.8b\n"         /* outr00 += 12345678 * w02 */

                    "ext v4.8b, v21.8b, v2.8b, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v5.8b, v2.8b, v3.8b, #1       \n" /* vext_s8(vinr0, vinr0_1, 1); 12345678 */

                    //r1
                    "sub   %[din_ptr2], %[din_ptr2], #1                       \n"
                    "sub   %[din_ptr3], %[din_ptr3], #1                       \n"

                    "smull  v19.8h,  %[v1].8b,  v2.8b   \n"       /* outr10 += 01234567 * w11 */
                    "smlal  v18.8h,  %[v4].8b,  v2.8b   \n"       /* outr00 += 01234567 * w11 */

                    "ext v14.8b, v21.8b, v6.8b, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v15.8b, v6.8b, v7.8b, #1       \n" /* vext_s8(vinr0, vinr0_1, 1); 12345678 */

                    "smlal  v19.8h,  %[v0].8b,  v4.8b   \n"       /* outr00 += 01234567 * w11 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v3].8b,  v4.8b   \n"       /* outr00 += 001234567 * w10 */

                    "ld1    {v0.8b}, [%[din_ptr0]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8                       \n"        /* load a00-a015 to q0*/

                    "saddw   v12.4s, v12.4s, v19.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v2].8b,  v5.8b   \n"       /* outr00 += 01234567 * w11 */
                    "smlal  v18.8h,  %[v5].8b,  v5.8b   \n"       /* outr00 += 12345678 * w12 */

                    //r2
                    "ld1    {v1.8b}, [%[din_ptr0]]                   \n"        /* load a00-a015 to q0*/
                    "ld1    {v3.8b}, [%[din_ptr1]]                   \n"        /* load a00-a015 to q0*/

                    "smlal  v19.8h,  %[v4].8b,  v6.8b   \n"       /* outr10 += 01234567 * w11 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v7].8b,  v6.8b   \n"       /* outr00 += 01234567 * w11 */

                    "ext v4.8b, v21.8b, v8.8b, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v5.8b, v8.8b, v9.8b, #1       \n" /* vext_s8(vinr0, vinr0_1, 1); 12345678 */

                    "saddw   v12.4s, v12.4s, v19.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v3].8b,  v14.8b   \n"       /* outr10 += 01234567 * w11 */
                    "smlal  v18.8h,  %[v6].8b,  v14.8b   \n"       /* outr00 += 01234567 * w11 */

                    "ld1    {v6.8b}, [%[din_ptr2]], #8                       \n"        /* load a00-a015 to q0*/

                    "smlal  v19.8h,  %[v5].8b,  v15.8b   \n"       /* outr10 += 01234567 * w11 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v8].8b,  v15.8b   \n"       /* outr00 += 01234567 * w11 */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"             /* v11 += outr00.high*/

                    //r3
                    "smull  v19.8h,  %[v7].8b,  v8.8b   \n"       /* outr00 += 01234567 * w11 */

                    "ld1    {v8.8b}, [%[din_ptr3]], #8                       \n"        /* load a00-a015 to q0*/

                    "ld1    {v7.8b}, [%[din_ptr2]]                   \n"        /* load a00-a015 to q0*/
                    "ld1    {v9.8b}, [%[din_ptr3]]                   \n"        /* load a00-a015 to q0*/

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smlal  v19.8h,  %[v6].8b,  v4.8b     \n"       /* outr00 += 01234567 * w11 */

                    "smax  v10.4s, v10.4s, v21.4s        \n"       /* relu*/
                    "smax  v11.4s, v11.4s, v21.4s        \n"       /* relu*/

                    "stp     q10, q11, [%[ptr_out0]], #32 \n"      /* store q10, q11 -> ptr_out       */

                    "saddw   v12.4s, v12.4s, v19.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v8].8b,  v5.8b        \n"       /* outr00 += 01234567 * w11 */

                    "ld1    {v10.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v11.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    "saddw   v12.4s, v12.4s, v19.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"             /* v11 += outr00.high*/

                    "smax  v12.4s, v12.4s, v21.4s        \n"       /* relu*/
                    "smax  v13.4s, v13.4s, v21.4s        \n"       /* relu*/

                    "stp     q12, q13, [%[ptr_out1]], #32   \n"      /* store q10, q11 -> ptr_out       */

                    "ld1    {v12.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v13.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    "cmp  %[cnt], #1                \n"
                    "blt 3f                         \n"
                //mid
                    "1:                             \n"
                    "ext v4.8b, v0.8B, v1.8b, #1       \n" /*12345678 */
                    "ext v5.8b, v0.8b, v1.8B, #2       \n" /*23456789 */

                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ext v14.8b, v2.8B, v3.8b, #1       \n" /*12345678 */
                    "ext v15.8b, v2.8b, v3.8B, #2       \n" /*23456789 */

                    "smlal  v18.8h,  %[v1].8b,  v4.8b\n"         /* outr00 += 12345678 * w01 */

                    "ext v16.8b, v6.8B, v7.8b, #1       \n" /*12345678 */
                    "ext v17.8b, v6.8b, v7.8B, #2       \n" /*23456789 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v2].8b,  v5.8b\n"         /* outr00 += 23456789 * w02 */

                    //r1
                    "ext v4.8b, v8.8B, v9.8b, #1       \n" /*12345678 */
                    "ext v5.8b, v8.8b, v9.8B, #2       \n" /*23456789 */

                    "smull  v19.8h,  %[v0].8b,  v2.8b   \n"       /* outr00 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ld1    {v0.8b}, [%[din_ptr0]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8                       \n"        /* load a00-a015 to q0*/

                    "smlal  v19.8h,  %[v1].8b,  v14.8b\n"         /* outr00 += 12345678 * w01 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v4].8b,  v14.8b\n"         /* outr00 += 12345678 * w01 */

                    "ld1    {v1.8b}, [%[din_ptr0]]                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v3.8b}, [%[din_ptr1]]                       \n"        /* load a00-a015 to q0*/
                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v2].8b,  v15.8b\n"         /* outr00 += 23456789 * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b\n"         /* outr00 += 12345678 * w01 */

                    //r2
                    "smlal  v19.8h,  %[v3].8b,  v6.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v6].8b,  v6.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v4].8b,  v16.8b\n"         /* outr00 += 12345678 * w01 */
                    "smlal  v18.8h,  %[v7].8b,  v16.8b\n"         /* outr00 += 12345678 * w01 */

                    "smlal  v19.8h,  %[v5].8b,  v17.8b\n"         /* outr00 += 23456789 * w02 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v8].8b,  v17.8b\n"         /* outr00 += 12345678 * w01 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ld1    {v6.8b}, [%[din_ptr2]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v8.8b}, [%[din_ptr3]], #8                       \n"        /* load a00-a015 to q0*/

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smlal  v19.8h,  %[v7].8b,  v4.8b\n"         /* outr00 += 12345678 * w01 */

                    "ld1    {v7.8b}, [%[din_ptr2]]                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v9.8b}, [%[din_ptr3]]                       \n"        /* load a00-a015 to q0*/

                    "smax  v10.4s, v10.4s, v21.4s        \n"       /* relu*/
                    "smax  v11.4s, v11.4s, v21.4s        \n"       /* relu*/

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v8].8b,  v5.8b\n"         /* outr00 += 23456789 * w02 */

                    "stp     q10, q11, [%[ptr_out0]], #32 \n"      /* store q10, q11 -> ptr_out       */

                    "ld1    {v10.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v11.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "subs %[cnt], %[cnt], #1            \n"

                    "smax  v12.4s, v12.4s, v21.4s        \n"       /* relu*/
                    "smax  v13.4s, v13.4s, v21.4s        \n"       /* relu*/

                    "stp     q12, q13, [%[ptr_out1]], #32 \n"      /* store q10, q11 -> ptr_out       */

                    "ld1    {v12.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v13.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    "bne 1b                                 \n"
                //right
                    "3:                             \n"
                    "ld1 {v14.8b}, [%[vmask]], #8             \n"
                    "ld1 {v15.8b}, [%[vmask]]                \n"

                    "bif v0.8b, v21.8b, v14.8b               \n"
                    "bif v1.8b, v21.8b, v15.8b               \n"
                    "bif v2.8b, v21.8b, v14.8b               \n"
                    "bif v3.8b, v21.8b, v15.8b               \n"

                    "ext v4.8b, v0.8b, v1.8b, #1             \n"
                    "ext v5.8b, v0.8b, v1.8b, #2             \n"

                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ext v16.8b, v2.8b, v3.8b, #1             \n"
                    "ext v17.8b, v2.8b, v3.8b, #2             \n"

                    "bif v6.8b, v21.8b, v14.8b               \n"
                    "bif v7.8b, v21.8b, v15.8b               \n"

                    "smlal  v18.8h,  %[v1].8b,  v4.8b   \n"       /* outr00 = 01234567 * w00 */

                    "bif v8.8b, v21.8b, v14.8b               \n"
                    "bif v9.8b, v21.8b, v15.8b               \n"

                    "ext v20.8b, v6.8b, v7.8b, #1             \n"
                    "ext v22.8b, v6.8b, v7.8b, #2             \n"

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v2].8b,  v5.8b   \n"       /* outr00 = 01234567 * w00 */

                    //r1
                    "ext v4.8b, v8.8b, v9.8b, #1             \n"
                    "ext v5.8b, v8.8b, v9.8b, #2             \n"

                    "smull  v19.8h,  %[v0].8b,  v2.8b   \n"       /* outr00 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ld1 {v14.4s}, [%[rmask]], #16                \n"
                    "ld1 {v15.4s}, [%[rmask]]                     \n"

                    "smlal  v19.8h,  %[v1].8b,  v16.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v4].8b,  v16.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ld1 {v0.4s}, [%[ptr_out0]], #16                \n"
                    "ld1 {v2.4s}, [%[ptr_out1]], #16                \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v2].8b,  v17.8b   \n"       /* outr00 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v5].8b,  v17.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ld1 {v1.4s}, [%[ptr_out0]]                   \n"
                    "ld1 {v3.4s}, [%[ptr_out1]]                   \n"

                    //r2
                    "smlal  v19.8h,  %[v3].8b,  v6.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b   \n"       /* outr00 = 01234567 * w00 */

                    "sub %[ptr_out0], %[ptr_out0], #16   \n"
                    "sub %[ptr_out1], %[ptr_out1], #16   \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v4].8b,  v20.8b   \n"       /* outr00 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v7].8b,  v20.8b   \n"       /* outr00 = 01234567 * w00 */

                    "smlal  v19.8h,  %[v5].8b,  v22.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v8].8b,  v22.8b   \n"       /* outr00 = 01234567 * w00 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smlal  v19.8h,  %[v7].8b,  v4.8b   \n"       /* outr00 = 01234567 * w00 */

                    "smax  v10.4s, v10.4s, v21.4s        \n"       /* relu*/
                    "smax  v11.4s, v11.4s, v21.4s        \n"       /* relu*/

                    "bif v10.16b, v0.16b, v14.16b         \n"
                    "bif v11.16b, v1.16b, v15.16b         \n"

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v8].8b,  v5.8b   \n"       /* outr00 = 01234567 * w00 */

                    "stp     q10, q11, [%[ptr_out0]], #32 \n"      /* store q10, q11 -> ptr_out       */

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smax  v12.4s, v12.4s, v21.4s        \n"       /* relu*/
                    "smax  v13.4s, v13.4s, v21.4s        \n"       /* relu*/

                    "bif v12.16b, v2.16b, v14.16b         \n"
                    "bif v13.16b, v3.16b, v15.16b         \n"

                    "stp     q12, q13, [%[ptr_out1]], #32 \n"      /* store q10, q11 -> ptr_out       */

                    :[cnt]"+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [din_ptr3] "+r"(din_ptr3), [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
                        [vmask] "+r" (val_mask), [rmask] "+r" (rst_mask)
                    : [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), [bias_val] "r" (vbias), \
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22)
                    :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12","v13","v14","v15", "v16", "v17", "v18",\
                      "v19", "v20", "v21", "v22"
                );
#else
                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vmov.u32 d11, #0                   @ zero\n"
                    //out0
                    "vdup.32 q8, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                            @ and \n" //q9 = vbias
                    //out1
                    "vdup.32 q10, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                            @ and \n" //q9 = vbias

                    //r0
                    "vmull.s8 q12, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678

                    "vld1.8 {d12-d13}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vdup.s8     d5, d0[3]               @ d5 = w10, w10, w00, w00\n"
                    "vdup.s8     d6, d0[4]               @ d6 = w11, w11, w01, w01\n"

                    "vmlal.s8 q12, d30, d2                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                    "vdup.s8     d7, d0[5]               @ d7 = w12, w12\n"
                    "add %[din_ptr0], #7                   @add \n"
                    "add %[din_ptr1], #7                   @add \n"
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d3                 @ out1 = din1 * w01 \n" // q13 = d12 * w01

                    "vmlal.s8 q12, d12, d6                 @ out0 = din1 * w11 \n" // q12 = d12 * w11

                    "vld1.8 {d12-d13}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vdup.s8     d8, d0[6]               @ d8 = w20, w00, w00, w00\n"
                    "vdup.s8     d9, d0[7]               @ d9 = w21, w01, w01, w01\n"
                    "vdup.s8     d10, d1[0]               @ d10 = w22, w02, w02, w02\n"

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d30, d2                 @ out1 += din1 * w00 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d5                 @ out0 += din1 * w10 \n" // q12 += d10 * w00

                    "add %[din_ptr2], #7                   @add \n"
                    "add %[din_ptr3], #7                   @add \n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #1          @ ext \n" //d11 = 12345678

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d14, d6                 @ out1 = din2 * w11 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d9                 @ out1 = din2 * w21 \n" // q13 = d12 * w01

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d30, d5                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d8                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678
                    "vmov.u32 q0, #0                         @ mov \n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d12, d9                 @ out1 = din3 * w21 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "vmax.s32 q8, q8, q0              @ max \n"
                    "vmax.s32 q9, q9, q0              @ max \n"

                    "vmlal.s8 q13, d30, d8                 @ out1 += din3 * w20 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"

                    "vst1.32 {d16-d17}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmax.s32 q10, q10, q0              @ max \n"
                    "vmax.s32 q11, q11, q0              @ max \n"

                    "vst1.32 {d20-d21}, [%[dout_ptr2]]!         @ store\n"
                    "cmp %[cnt], #1                                 \n"
                    "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"
                    "blt 1f                                         \n"

                //mid
                    "2:                                          \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    //out0
                    "vdup.32 q8, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                            @ and \n" //q9 = vbias
                    //out1
                    "vdup.32 q10, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                            @ and \n" //q9 = vbias

                     //r0
                    "vmull.s8 q12, d12, d2                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vext.8     d30, d12, d13, #1     @ ext \n" //d10 = 12345678
                    "vext.8     d31, d12, d13, #2          @ ext \n" //d11 = 23456789

                    "vld1.8 {d12-d13}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q12, d30, d3                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                    "add %[din_ptr0], #8                   @add \n"
                    "add %[din_ptr1], #8                   @add \n"
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8     d30, d12, d13, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          @ ext \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d2                 @ out1 = din1 * w01 \n" // q13 = d12 * w01

                    "vmlal.s8 q12, d12, d5                 @ out0 = din1 * w11 \n" // q12 = d12 * w11

                    "vld1.8 {d12-d13}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d30, d3                 @ out1 += din1 * w00 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q12, d30, d6                 @ out0 += din1 * w10 \n" // q12 += d10 * w00

                    "add %[din_ptr2], #8                   @add \n"
                    "add %[din_ptr3], #8                   @add \n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 12345678

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d14, d5                 @ out1 = din2 * w11 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d8                 @ out1 = din2 * w21 \n" // q13 = d12 * w01

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d30, d6                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d12, d13, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d12, d8                 @ out1 = din3 * w21 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "vmax.s32 q8, q8, q0              @ max \n"
                    "vmax.s32 q9, q9, q0              @ max \n"

                    "vmlal.s8 q13, d30, d9                 @ out1 += din3 * w20 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"

                    "vst1.32 {d16-d17}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmax.s32 q10, q10, q0              @ max \n"
                    "vmax.s32 q11, q11, q0              @ max \n"

                    "vst1.32 {d20-d21}, [%[dout_ptr2]]!         @ store\n"
                    "subs %[cnt], #1                                \n"
                    "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"
                    "bne  2b                                        \n"
                //right
                    "1:                                          \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    //out0
                    "vdup.32 q8, %[bias]                 @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                 @ and \n" //q9 = vbias
                    //out1
                    "vdup.32 q10, %[bias]                @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                @ and \n" //q9 = vbias

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"
                    "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                     //r0
                    "vmull.s8 q12, d12, d2                 @ out0 = din0 * w00 \n" // q12 = d12 * w01
                    "vext.8 d30, d12, d13, #1               @ ext \n" //d10 = 12345678
                    "vext.8 d31, d12, d13, #2               @ ext \n" //d11 = 23456789

                    "vld1.8 {d12-d13}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                    "vmlal.s8 q12, d30, d3                 @ out0 += din0 * w01 \n" // q12 += d10 * w00

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8 d30, d14, d15, #1           @ ext \n" //d10 = 00123456
                    "vext.8 d31, d14, d15, #2          @ ext \n" //d11 = 12345678

                    "vmull.s8 q13, d14, d2                 @ out1 = din1 * w00 \n" // q13 = d12 * w01

                    "vmlal.s8 q12, d14, d5                 @ out0 = din1 * w10 \n" // q12 = d12 * w11

                    "vld1.8 {d14-d15}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vbif.8 d12, d11, d28                 @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29                 @ bit select, deal with right pad\n"

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d30, d3                 @ out1 += din1 * w01 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d6                 @ out0 += din1 * w11 \n" // q12 += d10 * w00

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8 d30, d12, d13, #1               @ ext \n" //d10 = 00123456
                    "vext.8 d31, d12, d13, #2               @ ext \n" //d11 = 12345678

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d12, d5                 @ out1 = din2 * w10 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d12, d8                 @ out1 = din2 * w20 \n" // q13 = d12 * w01

                    "vbif.8 d14, d11, d28                     @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29                     @ bit select, deal with right pad\n"

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d30, d6                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    "vld1.32 {d28-d29}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d12-d13}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d14, d8                 @ out1 = din3 * w20 \n" // q13 = d12 * w01
                    "vld1.32 {d14-d15}, [%[dout_ptr2]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d24-d25}, [%[dout_ptr2]]     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vmax.s32 q8, q8, q0              @ max \n"
                    "vmax.s32 q9, q9, q0              @ max \n"

                    "vmlal.s8 q13, d30, d9                 @ out1 += din3 * w21 \n" // q13 += d10 * w00
                    "vbif q8, q14, q1                   @ bit select, deal with right pad\n"
                    "vbif q9, q6, q2                    @ bit select, deal with right pad\n"
                    "sub %[dout_ptr1], #16                  @ sub \n"
                    "sub %[dout_ptr2], #16                  @ sub \n"

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    "vst1.32 {d16-d17}, [%[dout_ptr1]]!         @ store\n"
                    "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmax.s32 q10, q10, q0              @ max \n"
                    "vmax.s32 q11, q11, q0              @ max \n"

                    "vbif q10, q7, q1        @ bit select, deal with right pad\n"
                    "vbif q11, q12, q2       @ bit select, deal with right pad\n"

                    "vst1.32 {d20-d21}, [%[dout_ptr2]]!         @ store\n"
                    "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2), \
                      [din_ptr3] "+r" (din_ptr3), \
                      [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                      "q11", "q12", "q13", "q14", "q15"
                );
#endif
                dout_ptr += 2 * w_out;
            }
        }
    }
    // write_out
    int32_to_dtype<Dtype>(pre_out, dout, scale, ch_in, num, h_out * w_out);
}

template <typename Dtype>
void conv_depthwise_3x3s1p1_bias_s_relu_int8(Dtype* dout,
                                      const int8_t* din,
                                      const int8_t* weights,
                                      const float* scale,
                                      const float* bias,
                                      bool flag_bias,
                                      int num,
                                      int ch_in,
                                      int h_in,
                                      int w_in,
                                      int h_out,
                                      int w_out,
                                      ARMContext* ctx) {
    //! pad is done implicit
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const uint8_t right_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    int8_t* zero_ptr = ctx->workspace_data<int8_t>();
    memset(zero_ptr, 0, w_in * sizeof(int8_t));
    int* write_ptr = reinterpret_cast<int*>(zero_ptr) + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_h = (h_out + 3) >> 2;

    unsigned int size_pad_right = (unsigned int)(w_in);

    int size_pad_bottom = h_out % 4;

    uint8x8_t vmask_rp = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    unsigned int rst_remain = (unsigned int)w_out;
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    uint8_t vmask[8];
    vst1_u8(vmask, vmask_rp);

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);

    int8x8_t vzero = vdup_n_s8(0);
    int32x4_t vzero_32 = vdupq_n_s32(0);
    int32_t* pre_out = reinterpret_cast<int*>(write_ptr + w_out + 4);

    for (int n = 0; n < num; ++n) {
        const int8_t *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = pre_out + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const int8_t* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? static_cast<int>(bias[c] / scale[c]) : 0;

            const int8_t* wei_ptr = weights + c * w_stride;
#ifdef __aarch64__
            int vbias[4] = {bias_val, bias_val, bias_val, bias_val};
            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);
#endif

            int *doutr0 = nullptr;
            int *doutr1 = nullptr;

            const int8_t *dr0 = din_ch_ptr;
            const int8_t *dr1 = dr0 + w_in;
            const int8_t *dr2 = dr1 + w_in;
            const int8_t *dr3 = dr2 + w_in;

            const int8_t *din_ptr0 = nullptr;
            const int8_t *din_ptr1 = nullptr;
            const int8_t *din_ptr2 = nullptr;
            const int8_t *din_ptr3 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                int out_buf1[8];
                int out_buf2[8];
                int trash_buf[8];

                unsigned int* rst_mask = rmask;
                uint8_t* val_mask = vmask;

                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    din_ptr3 = dr2;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr3;
                    dr3 = dr2 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr3;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                }
                //! process bottom pad
                if (i + 3 > h_in) {
                    switch (i + 3 - h_in) {
                        case 3:
                            din_ptr1 = zero_ptr;
                        case 2:
                            din_ptr2 = zero_ptr;
                        case 1:
                            din_ptr3 = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 2 > h_out) {
                    doutr1 = trash_buf;
                }
#ifdef __aarch64__
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr3]] \n"
                    "movi   v21.4s, #0x0\n"                         /* out0 = 0 */
                // left
                    "ld1 {v4.8b}, [%[vmask]]            \n"
                    "ld1    {v0.8b}, [%[din_ptr0]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v1.8b}, [%[din_ptr1]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v2.8b}, [%[din_ptr2]], #8                       \n"        /* load a00-a015 to q0*/
                    "ld1    {v3.8b}, [%[din_ptr3]], #8                       \n"        /* load a00-a015 to q0*/

                    "bif v0.8b, v21.8b, v4.8b               \n"
                    "bif v1.8b, v21.8b, v4.8b               \n"
                    "bif v2.8b, v21.8b, v4.8b               \n"
                    "bif v3.8b, v21.8b, v4.8b               \n"

                    "ext v6.8b, v21.8b, v0.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v7.8b, v0.8b, v21.8B, #1       \n" /* vext_s8(vinr0, vinr0_1, 1); 12345678 */

                    "ld1 {v10.4s}, [%[vbias]]            \n"
                    "ld1 {v11.4s}, [%[vbias]]            \n"

                    //r0
                    "smull  v18.8h,  %[v1].8b,  v0.8b   \n"       /* outr00 = 01234567 * w01 */

                    "ext v8.8b, v21.8b, v1.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v9.8b, v1.8b, v21.8B, #1       \n" /* vext_s8(vinr0, vinr0_1, 1); 12345678 */

                    "smlal  v18.8h,  %[v0].8b,  v6.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ld1 {v12.4s}, [%[vbias]]            \n"
                    "ld1 {v13.4s}, [%[vbias]]            \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v2].8b,  v7.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ext v6.8b, v21.8b, v2.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v7.8b, v2.8b, v21.8B, #1       \n" /* vext_s8(vinr0, vinr0_1, 1); 12345678 */

                    //r1
                    "smull  v19.8h,  %[v1].8b,  v1.8b   \n"       /* outr00 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v4].8b,  v1.8b   \n"       /* outr00 = 01234567 * w00 */

                    // "ld1 {v14.4s}, [%[rmask]], #16                \n"
                    // "ld1 {v15.4s}, [%[rmask]]                     \n"

                    "smlal  v19.8h,  %[v0].8b,  v8.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v3].8b,  v8.8b   \n"       /* outr00 = 01234567 * w00 */

                    // "ld1 {v16.4s}, [%[ptr_out0]], #16                \n"
                    // "ld1 {v17.4s}, [%[ptr_out1]], #16                \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v2].8b,  v9.8b   \n"       /* outr00 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v5].8b,  v9.8b   \n"       /* outr00 = 01234567 * w00 */

                    "ext v8.8b, v21.8b, v3.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 00123456 */
                    "ext v9.8b, v3.8b, v21.8B, #1       \n"  //vext_s8(vinr0, vinr0_1, 1); 12345678

                    // "ld1 {v0.4s}, [%[ptr_out0]]                   \n"
                    // "ld1 {v1.4s}, [%[ptr_out1]]                   \n"

                    //r2
                    "smlal  v19.8h,  %[v4].8b,  v2.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v7].8b,  v2.8b   \n"       /* outr00 = 01234567 * w00 */

                    // "sub %[ptr_out0], %[ptr_out0], #16   \n"
                    // "sub %[ptr_out1], %[ptr_out1], #16   \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v3].8b,  v6.8b   \n"       /* outr00 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v6].8b,  v6.8b   \n"       /* outr00 = 01234567 * w00 */

                    "smlal  v19.8h,  %[v5].8b,  v7.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smull  v18.8h,  %[v8].8b,  v7.8b   \n"       /* outr00 = 01234567 * w00 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    //r3
                    "smull  v19.8h,  %[v7].8b,  v3.8b   \n"       /* outr00 = 01234567 * w00 */

                    "saddw   v10.4s, v10.4s, v18.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"             /* v11 += outr00.high*/

                    "smlal  v19.8h,  %[v6].8b,  v8.8b   \n"       /* outr00 = 01234567 * w00 */

                    "smax  v10.4s, v10.4s, v21.4s       \n"      /* relu */
                    "smax  v11.4s, v11.4s, v21.4s       \n"      /* relu */

                    // "bif v10.16b, v16.16b, v14.16b         \n"
                    // "bif v11.16b, v0.16b, v15.16b         \n"

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smull  v19.8h,  %[v8].8b,  v9.8b   \n"       /* outr00 = 01234567 * w00 */

                    "stp     q10, q11, [%[ptr_out0]] \n"      /* store q10, q11 -> ptr_out       */

                    "saddw   v12.4s, v12.4s, v19.4h     \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"             /* v11 += outr00.high*/

                    "smax  v12.4s, v12.4s, v21.4s       \n"      /* relu */
                    "smax  v13.4s, v13.4s, v21.4s       \n"      /* relu */

                    // "bif v12.16b, v17.16b, v14.16b         \n"
                    // "bif v13.16b, v1.16b, v15.16b         \n"

                    "stp     q12, q13, [%[ptr_out1]] \n"      /* store q10, q11 -> ptr_out       */

                    :[din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [din_ptr3] "+r"(din_ptr3), \
                        [rmask] "+r" (rst_mask)
                    : [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), [vbias] "r" (vbias), \
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22), [vmask] "r" (vmask), \
                         [ptr_out0] "r"(out_buf1), [ptr_out1] "r"(out_buf2)
                    :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12","v13","v14","v15", "v16", "v17", "v18",\
                      "v19", "v20", "v21", "v22"
                );
#else
                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"
                    "vld1.8 {d28}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d12}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d13}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"

                    "vmov.u32 d11, #0                   @ zero\n"
                    //out0
                    "vdup.32 q8, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                            @ and \n" //q9 = vbias

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d28        @ bit select, deal with right pad\n"
                    "vld1.8 {d14}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d15}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    //out1
                    "vdup.32 q10, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                            @ and \n" //q9 = vbias

                    //r0
                    "vmull.s8 q12, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vext.8 d30, d11, d12, #7           @ ext \n" //d10 = 00123456
                    "vext.8 d31, d12, d11, #1          @ ext \n" //d11 = 12345678

                    "vdup.s8 d5, d0[3]               @ d5 = w10, w10, w00, w00\n"
                    "vdup.s8 d6, d0[4]               @ d6 = w11, w11, w01, w01\n"

                    "vmlal.s8 q12, d30, d2                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                    "vdup.s8 d7, d0[5]               @ d7 = w12, w12\n"
                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d28        @ bit select, deal with right pad\n"
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8     d30, d11, d13, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d13, d11, #1          @ ext \n" //d11 = 12345678
                    "vmull.s8 q13, d13, d3                 @ out1 = din1 * w01 \n" // q13 = d12 * w01

                    "vmlal.s8 q12, d13, d6                 @ out0 = din1 * w11 \n" // q12 = d12 * w11

                    "vdup.s8 d8, d0[6]               @ d8 = w20, w00, w00, w00\n"
                    "vdup.s8 d9, d0[7]               @ d9 = w21, w01, w01, w01\n"

                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d30, d2                 @ out1 += din1 * w00 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d5                 @ out0 += din1 * w10 \n" // q12 += d10 * w00

                    "vdup.s8 d10, d1[0]               @ d10 = w22, w02, w02, w02\n"
                    // "vld1.32 {d28-d29}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    // "vld1.32 {d12-d13}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d11, #1          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d14, d6                 @ out1 = din2 * w11 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d9                 @ out1 = din2 * w21 \n" // q13 = d12 * w01

                    // "sub %[dout_ptr1], #16                  @ sub \n"

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d30, d5                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d8                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    // "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    // "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d11, d15, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d15, d11, #1          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d15, d9                 @ out1 = din3 * w21 \n" // q13 = d12 * w01

                    "vmov.u32 q0, #0                   @ zero\n"

                    // "vld1.32 {d6-d7}, [%[dout_ptr2]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    // "vld1.32 {d14-d15}, [%[dout_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d30, d8                 @ out1 += din3 * w20 \n" // q13 += d10 * w00

                    "vmax.s32 q8, q8, q0                    @ max \n"
                    "vmax.s32 q9, q9, q0                    @ max \n"

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmull.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    // "sub %[dout_ptr2], #16                  @ sub \n"
                    // "vbif q8, q14, q1                   @ bit select, deal with right pad\n"
                    // "vbif q9, q6, q2                    @ bit select, deal with right pad\n"

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vst1.32 {d16-d19}, [%[dout_ptr1]]         @ store\n"
                    // "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"

                    "vmax.s32 q10, q10, q0                    @ max \n"
                    "vmax.s32 q11, q11, q0                    @ max \n"

                    // "vbif q10, q3, q1                   @ bit select, deal with right pad\n"
                    // "vbif q11, q7, q2                    @ bit select, deal with right pad\n"

                    "vst1.32 {d20-d23}, [%[dout_ptr2]]         @ store\n"
                    // "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2), \
                      [din_ptr3] "+r" (din_ptr3), \
                      [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask), \
                      [dout_ptr1] "r"(out_buf1), [dout_ptr2] "r"(out_buf2)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                      "q11", "q12", "q13", "q14", "q15"
                );
#endif
                for (int w = 0; w < w_out; ++w){
                    *doutr0++ = out_buf1[w];
                    *doutr1++ = out_buf2[w];
                }
                dout_ptr += 2 * w_out;
            }
        }
    }
    // write_out
    int32_to_dtype<Dtype>(pre_out, dout, scale, ch_in, num, h_out * w_out);
}

template void conv_depthwise_3x3s1p1_int8<int8_t>(int8_t* dout,
                                                  const int8_t* din,
                                                  const int8_t* weights,
                                                  const float* scale,
                                                  const float* bias,
                                                  bool flag_bias,
                                                  bool flag_act,
                                                  int num,
                                                  int chin,
                                                  int hin,
                                                  int win,
                                                  int hout,
                                                  int wout,
                                                  ARMContext* ctx);

template void conv_depthwise_3x3s1p1_int8<float>(float* dout,
                                                  const int8_t* din,
                                                  const int8_t* weights,
                                                  const float* scale,
                                                  const float* bias,
                                                  bool flag_bias,
                                                  bool flag_act,
                                                  int num,
                                                  int chin,
                                                  int hin,
                                                  int win,
                                                  int hout,
                                                  int wout,
                                                  ARMContext* ctx);

template void conv_depthwise_3x3s1p1_bias_int8<int8_t>(int8_t* dout,
                                                       const int8_t* din,
                                                       const int8_t* weights,
                                                       const float* scale,
                                                       const float* bias,
                                                       bool flag_bias,
                                                       int num,
                                                       int chin,
                                                       int hin,
                                                       int win,
                                                       int hout,
                                                       int wout,
                                                       ARMContext* ctx);

template void conv_depthwise_3x3s1p1_bias_int8<float>(float* dout,
                                                  const int8_t* din,
                                                  const int8_t* weights,
                                                  const float* scale,
                                                  const float* bias,
                                                  bool flag_bias,
                                                  int num,
                                                  int chin,
                                                  int hin,
                                                  int win,
                                                  int hout,
                                                  int wout,
                                                  ARMContext* ctx);

template void conv_depthwise_3x3s1p1_bias_s_int8<int8_t>(int8_t* dout,
                                                         const int8_t* din,
                                                         const int8_t* weights,
                                                         const float* scale,
                                                         const float* bias,
                                                         bool flag_bias,
                                                         int num,
                                                         int chin,
                                                         int hin,
                                                         int win,
                                                         int hout,
                                                         int wout,
                                                         ARMContext* ctx);

template void conv_depthwise_3x3s1p1_bias_s_int8<float>(float* dout,
                                                        const int8_t* din,
                                                        const int8_t* weights,
                                                        const float* scale,
                                                        const float* bias,
                                                        bool flag_bias,
                                                        int num,
                                                        int chin,
                                                        int hin,
                                                        int win,
                                                        int hout,
                                                        int wout,
                                                        ARMContext* ctx);

template void conv_depthwise_3x3s1p1_bias_relu_int8<int8_t>(int8_t* dout,
                                                            const int8_t* din,
                                                            const int8_t* weights,
                                                            const float* scale,
                                                            const float* bias,
                                                            bool flag_bias,
                                                            int num,
                                                            int chin,
                                                            int hin,
                                                            int win,
                                                            int hout,
                                                            int wout,
                                                                ARMContext* ctx);

template void conv_depthwise_3x3s1p1_bias_relu_int8<float>(float* dout,
                                                           const int8_t* din,
                                                           const int8_t* weights,
                                                           const float* scale,
                                                           const float* bias,
                                                           bool flag_bias,
                                                           int num,
                                                           int chin,
                                                           int hin,
                                                           int win,
                                                           int hout,
                                                           int wout,
                                                           ARMContext* ctx);

template void conv_depthwise_3x3s1p1_bias_s_relu_int8<int8_t>(int8_t* dout,
                                                              const int8_t* din,
                                                              const int8_t* weights,
                                                              const float* scale,
                                                              const float* bias,
                                                              bool flag_bias,
                                                              int num,
                                                              int chin,
                                                              int hin,
                                                              int win,
                                                              int hout,
                                                              int wout,
                                                              ARMContext* ctx);

template void conv_depthwise_3x3s1p1_bias_s_relu_int8<float>(float* dout,
                                                             const int8_t* din,
                                                             const int8_t* weights,
                                                             const float* scale,
                                                             const float* bias,
                                                             bool flag_bias,
                                                             int num,
                                                             int chin,
                                                             int hin,
                                                             int win,
                                                             int hout,
                                                             int wout,
                                                             ARMContext* ctx);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
