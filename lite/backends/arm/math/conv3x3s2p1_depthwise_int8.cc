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
void conv_depthwise_3x3s2p1_bias_int8(Dtype* dout,
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
void conv_depthwise_3x3s2p1_bias_s_int8(Dtype* dout,
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
void conv_depthwise_3x3s2p1_bias_relu_int8(Dtype* dout,
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
void conv_depthwise_3x3s2p1_bias_s_relu_int8(Dtype* dout,
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
void conv_depthwise_3x3s2p1_int8(Dtype* dout,
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
    if (win > 16) {
      conv_depthwise_3x3s2p1_bias_relu_int8(dout, din, weights, scale, bias, flag_bias, num, chin, hin, win, hout, wout, ctx);
    } else {
      conv_depthwise_3x3s2p1_bias_s_relu_int8(dout, din, weights, scale, bias, flag_bias, num, chin, hin, win, hout, wout, ctx);
    }
  } else {
    if (win > 16) {
      conv_depthwise_3x3s2p1_bias_int8(dout, din, weights, scale, bias, flag_bias, num, chin, hin, win, hout, wout, ctx);
    } else {
      conv_depthwise_3x3s2p1_bias_s_int8(dout, din, weights, scale, bias, flag_bias, num, chin, hin, win, hout, wout, ctx);
    }
  }
}

/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias, width > 4
 */
template <typename Dtype>
void conv_depthwise_3x3s2p1_bias_int8(Dtype* dout,
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
    const uint8_t right_pad_idx[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    int8_t* zero_ptr = ctx->workspace_data<int8_t>();
    memset(zero_ptr, 0, w_in * sizeof(int8_t));
    int* write_ptr = reinterpret_cast<int*>(zero_ptr) + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = (w_in + 15) >> 4;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(w_in - 15 - (cnt_col << 4));
    if (size_pad_right == 17){
        size_pad_right = 0;
        cnt_col++;
    }

    uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned int rst_remain = (unsigned int)(w_out - ((cnt_col + 1) << 3));
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    uint8x16_t vmask_rp = vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
    uint8_t vmask[16];
    vst1q_u8(vmask, vmask_rp);

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);

    int8x8_t vzero = vdup_n_s8(0);
    int32_t* pre_out = reinterpret_cast<int*>(write_ptr + w_out + 4);
    // printf("cnt_col: %d, rst_remain: %d, size_pad_right: %d\n", cnt_col, rst_remain, size_pad_right);
     for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = pre_out + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? static_cast<int>(bias[c] / scale[c]) : 0;

            const signed char* wei_ptr = weights + c * w_stride;
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

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;

                doutr0 = dout_ptr;
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr1 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr0 + w_in;
                    dr2 = dr1 + w_in;
                }
                //! process bottom pad
                if (i + 2 > h_in) {
                    switch (i + 2 - h_in) {
                        case 2:
                            din_ptr1 = zero_ptr;
                        case 1:
                            din_ptr2 = zero_ptr;
                        default:
                            break;
                    }
                }
#ifdef __aarch64__
                int cnt = cnt_col;
                uint8_t *val_mask = vmask;
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "movi   v10.4s, #0x0\n"
                // left
                    "ld2    {v0.8b - v1.8b}, [%[din_ptr0]]         \n"         /*load a00-a015 to q0*/
                    "ld2    {v2.8b - v3.8b}, [%[din_ptr1]]         \n"        /* load a00-a015 to q0*/
                    "ld2    {v4.8b - v5.8b}, [%[din_ptr2]]         \n"         /*load a00-a015 to q0*/

                    "ld1    {v12.4s}, [%[bias_val]] \n"                    /* dup v10, bias*/
                    "ld1    {v13.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    "ext v6.8b, v10.8b, v1.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 013579 */
                    "ext v7.8b, v10.8b, v3.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 013579 */
                    "ext v8.8b, v10.8b, v5.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 013579 */

                    //r0
                    "smull  v14.8h,  %[v1].8b,  v0.8b   \n"       /* outr00 = 02468 * w01 */
                    "smull  v15.8h,  %[v2].8b,  v1.8b\n"         /* outr00 += 13579 * w02 */
                    "smull  v16.8h,  %[v0].8b,  v6.8b\n"         /* outr00 += 013579 * w00 */

                    "add   %[din_ptr0], %[din_ptr0], #15                       \n"
                    "add   %[din_ptr1], %[din_ptr1], #15                       \n"
                    "add   %[din_ptr2], %[din_ptr2], #15                       \n"

                    //r1
                    "smlal  v14.8h,  %[v4].8b,  v2.8b   \n"       /* outr00 = 02468 * w01 */
                    "smlal  v15.8h,  %[v5].8b,  v3.8b\n"         /* outr00 += 13579 * w02 */
                    "smlal  v16.8h,  %[v3].8b,  v7.8b\n"         /* outr00 += 013579 * w00 */

                    "saddw   v12.4s, v12.4s, v14.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v14.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v15.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v15.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v16.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v16.8h        \n"             /* v11 += outr00.high*/

                    //r2
                    "smull  v14.8h,  %[v7].8b,  v4.8b   \n"       /* outr00 = 02468 * w01 */
                    "smull  v15.8h,  %[v8].8b,  v5.8b\n"         /* outr00 += 13579 * w02 */
                    "smull  v16.8h,  %[v6].8b,  v8.8b\n"         /* outr00 += 013579 * w00 */

                    "ld2    {v0.8b - v1.8b}, [%[din_ptr0]], #16         \n"         /*load a00-a015 to q0*/
                    "ld2    {v2.8b - v3.8b}, [%[din_ptr1]], #16         \n"        /* load a00-a015 to q0*/
                    "ld2    {v4.8b - v5.8b}, [%[din_ptr2]], #16         \n"         /*load a00-a015 to q0*/

                    "saddw   v12.4s, v12.4s, v14.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v14.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v15.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v15.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v16.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v16.8h        \n"             /* v11 += outr00.high*/

                    "stp     q12, q13, [%[ptr_out0]], #32   \n"      /* store q10, q11 -> ptr_out   */

                    "ld1    {v12.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v13.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    "cmp  %[cnt], #1                \n"
                    "blt 3f                         \n"
                //mid
                    "1:                             \n"
                    "ld1    {v6.8b}, [%[din_ptr0]]         \n"         /*load a00-a015 to q0*/
                    "ld1    {v7.8b}, [%[din_ptr1]]         \n"         /*load a00-a015 to q0*/
                    "ld1    {v8.8b}, [%[din_ptr2]]         \n"         /*load a00-a015 to q0*/

                    "ext v9.8b, v0.8b, v6.8B, #1       \n" /* vext_s8(vzero, vinr0, 7); 246810 */
                    "ext v11.8b, v2.8b, v7.8B, #1       \n" /* vext_s8(vzero, vinr0, 7); 246810 */
                    "ext v14.8b, v4.8b, v8.8B, #1       \n" /* vext_s8(vzero, vinr0, 7); 246810 */

                     //r0
                    "smull  v6.8h,  %[v0].8b,  v0.8b   \n"       /* outr00 = 02468 * w00 */
                    "smull  v7.8h,  %[v1].8b,  v1.8b\n"         /* outr00 += 13579 * w01 */
                    "smull  v8.8h,  %[v2].8b,  v9.8b\n"         /* outr00 += 246810 * w02 */

                    //r1
                    "smlal  v6.8h,  %[v3].8b,  v2.8b   \n"       /* outr00 = 02468 * w00 */
                    "smlal  v7.8h,  %[v4].8b,  v3.8b\n"         /* outr00 += 13579 * w01 */
                    "smlal  v8.8h,  %[v5].8b,  v11.8b\n"         /* outr00 += 246810 * w02 */

                    "saddw   v12.4s, v12.4s, v6.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v6.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v7.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v7.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v8.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v8.8h        \n"             /* v11 += outr00.high*/

                    //r2
                    "smull  v6.8h,  %[v6].8b,  v4.8b   \n"       /* outr00 = 02468 * w00 */
                    "smull  v7.8h,  %[v7].8b,  v5.8b\n"         /* outr00 += 13579 * w01 */
                    "smull  v8.8h,  %[v8].8b,  v14.8b\n"         /* outr00 += 246810 * w02 */

                    "ld2    {v0.8b - v1.8b}, [%[din_ptr0]], #16         \n"         /*load a00-a015 to q0*/
                    "ld2    {v2.8b - v3.8b}, [%[din_ptr1]], #16         \n"        /* load a00-a015 to q0*/
                    "ld2    {v4.8b - v5.8b}, [%[din_ptr2]], #16         \n"         /*load a00-a015 to q0*/

                    "saddw   v12.4s, v12.4s, v6.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v6.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v7.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v7.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v8.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v8.8h        \n"             /* v11 += outr00.high*/

                    "subs %[cnt], %[cnt], #1               \n"

                    "stp     q12, q13, [%[ptr_out0]], #32   \n"      /* store q10, q11 -> ptr_out   */

                    "ld1    {v12.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v13.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "bne 1b                         \n"
                //right
                    "3:                             \n"
                    "ld1 {v14.8b}, [%[vmask]], #8             \n"
                    "ld1 {v15.8b}, [%[vmask]]                \n"

                    "bif v0.8b, v10.8b, v14.8b               \n"
                    "bif v1.8b, v10.8b, v15.8b               \n"
                    "bif v2.8b, v10.8b, v14.8b               \n"
                    "bif v3.8b, v10.8b, v15.8b               \n"
                    "bif v4.8b, v10.8b, v14.8b               \n"
                    "bif v5.8b, v10.8b, v15.8b               \n"

                    "ext v6.8b, v0.8b, v10.8B, #1       \n" /* vext_s8(vzero, vinr0, 7); 2468.. */
                    "ext v7.8b, v2.8b, v10.8B, #1       \n" /* vext_s8(vzero, vinr0, 7); 2468..*/
                    "ext v8.8b, v4.8b, v10.8B, #1       \n" /* vext_s8(vzero, vinr0, 7); 2468.. */

                    //r0
                    "smull  v14.8h,  %[v0].8b,  v0.8b   \n"       /* outr00 = 02468 * w00 */
                    "smull  v15.8h,  %[v1].8b,  v1.8b\n"         /* outr00 += 13579 * w01 */
                    "smull  v16.8h,  %[v2].8b,  v6.8b\n"         /* outr00 += 246810 * w02 */

                    //r1
                    "smlal  v14.8h,  %[v3].8b,  v2.8b   \n"       /* outr00 = 02468 * w00 */
                    "smlal  v15.8h,  %[v4].8b,  v3.8b\n"         /* outr00 += 13579 * w01 */
                    "smlal  v16.8h,  %[v5].8b,  v7.8b\n"         /* outr00 += 246810 * w02 */

                    "saddw   v12.4s, v12.4s, v14.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v14.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v15.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v15.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v16.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v16.8h        \n"             /* v11 += outr00.high*/

                    //r2
                    "smull  v14.8h,  %[v6].8b,  v4.8b   \n"       /* outr00 = 02468 * w00 */
                    "smull  v15.8h,  %[v7].8b,  v5.8b\n"         /* outr00 += 13579 * w01 */
                    "smull  v16.8h,  %[v8].8b,  v8.8b\n"         /* outr00 += 246810 * w02 */

                    "ldp    q0, q1, [%[ptr_out0]] \n"                    /* dup v10, bias */
                    "ldp    q9, q11, [%[rst_mask]] \n"                    /* dup v10, bias */

                    "saddw   v12.4s, v12.4s, v14.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v14.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v15.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v15.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v16.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v16.8h        \n"             /* v11 += outr00.high*/

                    "bif v12.16b, v0.16b, v9.16b         \n"
                    "bif v13.16b, v1.16b, v11.16b         \n"

                    "stp     q12, q13, [%[ptr_out0]], #32 \n"      /* store q10, q11 -> ptr_out       */

                    :[cnt]"+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [ptr_out0] "+r"(doutr0), [vmask] "+r" (val_mask)
                    : [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), [bias_val] "r" (vbias), \
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22), [rst_mask] "r" (rmask)
                    :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", \
                      "v10", "v11", "v12","v13","v14","v15", "v16"
                );
#else
                unsigned int* rst_mask = rmask;
                int cnt = cnt_col;
                //prefetch input
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
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6
                    "vmov.u32 d11, #0                   @ zero\n"

                    "vdup.s8     d5, d0[3]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d6, d0[4]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d7, d0[5]               @ d4 = w02, w02, w02, w02\n"

                    "vext.8  d18, d11, d13, #7     @ ext \n" //d16 = -1 1 3 5
                    "vext.8  d19, d11, d15, #7     @ ext \n" //d17 = -1 1 3 5
                    "vext.8  d20, d11, d17, #7     @ ext \n" //d18 = -1 1 3 5

                    //r0
                    "vmull.s8 q13, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vmull.s8 q14, d13, d4                 @ out1 = din0 * w02 \n" // q12 = d12 * w02
                    "vmull.s8 q15, d18, d2                 @ out2 = din0 * w00 \n" // q12 = d12 * w02

                    "vdup.s8 d8, d0[6]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8 d9, d0[7]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8 d10, d1[0]               @ d4 = w02, w02, w02, w02\n"

                    //out0
                    "vdup.32 q11, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                            @ and \n" //q9 = vbias

                    //r1
                    "vmlal.s8 q13, d14, d6                 @ out0 += din1 * w11 \n" // q12 = d12 * w11
                    "vmlal.s8 q14, d15, d7                 @ out1 += din1 * w12 \n" // q12 = d12 * w11
                    "vmlal.s8 q15, d19, d5                 @ out2 += din1 * w10 \n" // q12 = d12 * w11

                    "add %[din_ptr0], #15                   @add \n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "add %[din_ptr1], #15                   @add \n"

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "add %[din_ptr2], #15                   @add \n"

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    //r2
                    "vmull.s8 q13, d16, d9                 @ out0 += din1 * w21 \n" // q12 = d12 * w11
                    "vmull.s8 q14, d17, d10                 @ out1 += din1 * w22 \n" // q12 = d12 * w11
                    "vmull.s8 q15, d20, d8                 @ out2 += din1 * w20 \n" // q12 = d12 * w11

                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vst1.32 {d22-d23}, [%[dout_ptr1]]!         @ store\n"
                    "cmp %[cnt], #1                                 \n"
                    "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    "blt 1f                                         \n"

                //mid
                    "2:                                              \n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]!    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]!    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]!    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6

                    "vld1.8 {d21}, [%[din_ptr0]]    @ load din00= 16 17\n" //d10 = 0 2 4 6
                    "vld1.8 {d22}, [%[din_ptr1]]    @ load din00= 16 17\n" //d12 = 0 2 4 6
                    "vld1.8 {d23}, [%[din_ptr2]]    @ load din00= 16 17\n" //d14 = 0 2 4 6

                    "vext.8  d18, d12, d21, #1     @ ext din00 = 2 4 6 8\n" //d16 = 2 4 6 8
                    "vext.8  d19, d14, d22, #1     @ ext \n" //d17 = 2 4 6 8
                    "vext.8  d20, d16, d23, #1     @ ext \n" //d18 = 2 4 6 8

                    //r0
                    "vmull.s8 q13, d12, d2                 @ out0 = din0 * w00 \n" // q12 = 0 2 4 6
                    "vmull.s8 q14, d13, d3                 @ out1 = din0 * w01 \n" // q12 = 1 3 5 7
                    "vmull.s8 q15, d18, d4                 @ out2 = din0 * w02 \n" // q12 = 2 4 6 8

                    //out0
                    "vdup.32 q11, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                            @ and \n" //q9 = vbias

                    //r1
                    "vmlal.s8 q13, d14, d5                 @ out0 += din1 * w10 \n" // q12 = 0 2 4 6
                    "vmlal.s8 q14, d15, d6                 @ out1 += din1 * w11 \n" // q12 = 1 3 5 7
                    "vmlal.s8 q15, d19, d7                 @ out2 += din1 * w12 \n" // q12 = 2 4 6 8

                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    //r2
                    "vmull.s8 q13, d16, d8                 @ out0 += din1 * w20 \n" // q12 = 0 2 4 6
                    "vmull.s8 q14, d17, d9                 @ out1 += din1 * w21 \n" // q12 = 1 3 5 7
                    "vmull.s8 q15, d20, d10                 @ out2 += din1 * w22 \n" // q12 = 2 4 6 8

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vst1.32 {d22-d23}, [%[dout_ptr1]]!         @ store\n"

                    "subs %[cnt], #1                                \n"
                    "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    "bne  2b                                        \n"
                //right
                    "1:                                              \n"
                    "cmp %[size_pad_right], #1                       \n"
                    "blt 3f                                         \n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]!    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]!    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]!    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6
                    "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    //out0
                    "vdup.32 q11, %[bias]                 @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                 @ and \n" //q9 = vbias

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                    "vext.8  d18, d12, d11, #1     @ ext din00 = 2 4 6 8\n" //d16 = -1 1 3 5
                    "vext.8  d19, d14, d11, #1     @ ext \n" //d17 = -1 1 3 5
                    "vext.8  d20, d16, d11, #1     @ ext \n" //d18 = -1 1 3 5

                    //r0
                    "vmull.s8 q13, d12, d2                 @ out0 = din0 * w00 \n" // q12 = 0 2 4 6
                    "vmull.s8 q14, d13, d3                 @ out1 = din0 * w01 \n" // q12 = 1 3 5 7
                    "vmull.s8 q15, d18, d4                 @ out2 = din0 * w02 \n" // q12 = 2 4 6 8

                    //r1
                    "vmlal.s8 q13, d14, d5                 @ out0 += din1 * w11 \n" // q12 = 0 2 4 6
                    "vmlal.s8 q14, d15, d6                 @ out1 += din1 * w12 \n" // q12 = 1 3 5 7
                    "vmlal.s8 q15, d19, d7                 @ out2 += din1 * w10 \n" // q12 = 2 4 6 8

                    "vld1.32 {d12-d13}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d14-d15}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "sub %[dout_ptr1], #16                  @ sub \n"

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    //r2
                    "vmull.s8 q13, d16, d8                 @ out0 += din1 * w11 \n" // q12 = 0 2 4 6
                    "vmull.s8 q14, d17, d9                 @ out1 += din1 * w12 \n" // q12 = 1 3 5 7
                    "vmull.s8 q15, d20, d10                 @ out2 += din1 * w10 \n" // q12 = 2 4 6 8

                    "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vbif q11, q6, q1        @ bit select, deal with right pad\n"
                    "vbif q12, q7, q2       @ bit select, deal with right pad\n"

                    "vst1.32 {d22-d23}, [%[dout_ptr1]]!         @ store\n"
                    "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    "3:                                             \n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [dout_ptr1] "+r"(doutr0), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask), [size_pad_right] "r" (size_pad_right)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"
                );
#endif
                dout_ptr += w_out;
            }
        }
    }
    // write_out
    int32_to_dtype<Dtype>(pre_out, dout, scale, ch_in, num, h_out * w_out);
}

//w_in <= 16
template <typename Dtype>
void conv_depthwise_3x3s2p1_bias_s_int8(Dtype* dout,
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
    // const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const uint8_t right_pad_idx[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    int8_t* zero_ptr = ctx->workspace_data<int8_t>();
    memset(zero_ptr, 0, w_in * sizeof(int8_t));
    int* write_ptr = reinterpret_cast<int*>(zero_ptr) + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    unsigned int size_pad_right = (unsigned int)(w_in);

    uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned int rst_remain = (unsigned int)w_out;
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    uint8x16_t vmask_rp = vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
    uint8_t vmask[16];
    vst1q_u8(vmask, vmask_rp);

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);

    int8x8_t vzero = vdup_n_s8(0);
    int32_t* pre_out = reinterpret_cast<int*>(write_ptr + w_out + 4);
    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = pre_out + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? static_cast<int>(bias[c] / scale[c]) : 0;

            const signed char* wei_ptr = weights + c * w_stride;
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

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;

                doutr0 = dout_ptr;
                int out_buf1[8];

                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr1 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr2 + w_in;
                    dr2 = dr1 + w_in;
                }
                //! process bottom pad
                if (i + 2 > h_in) {
                    switch (i + 2 - h_in) {
                        case 2:
                            din_ptr1 = zero_ptr;
                        case 1:
                            din_ptr2 = zero_ptr;
                        default:
                            break;
                    }
                }
#ifdef __aarch64__
                unsigned int* rst_mask = rmask;
                uint8_t* val_mask = vmask;
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "movi   v16.4s, #0x0\n"
                // left
                    "ld1 {v10.8b}, [%[vmask]], #8             \n"
                    "ld1 {v11.8b}, [%[vmask]]                \n"
                    "ld2    {v0.8b - v1.8b}, [%[din_ptr0]]         \n"         /*load a00-a015 to q0*/
                    "ld2    {v2.8b - v3.8b}, [%[din_ptr1]]         \n"        /* load a00-a015 to q0*/
                    "ld2    {v4.8b - v5.8b}, [%[din_ptr2]]         \n"         /*load a00-a015 to q0*/

                    "bif v0.8b, v16.8b, v10.8b               \n"
                    "bif v1.8b, v16.8b, v11.8b               \n"
                    "bif v2.8b, v16.8b, v10.8b               \n"
                    "bif v3.8b, v16.8b, v11.8b               \n"
                    "bif v4.8b, v16.8b, v10.8b               \n"
                    "bif v5.8b, v16.8b, v11.8b               \n"

                    "ld1    {v12.4s}, [%[bias_val]] \n"                    /* dup v10, bias*/
                    "ld1    {v13.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    "ext v6.8b, v16.8b, v1.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 013579 */
                    "ext v7.8b, v16.8b, v3.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 013579 */
                    "ext v8.8b, v16.8b, v5.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 013579 */

                    //r0
                    "smull  v17.8h,  %[v1].8b,  v0.8b   \n"       /* outr00 = 02468 * w01 */
                    "smull  v18.8h,  %[v2].8b,  v1.8b\n"         /* outr00 += 13579 * w02 */
                    "smull  v19.8h,  %[v0].8b,  v6.8b\n"         /* outr00 += 013579 * w00 */

                    // "ldp    q0, q1, [%[ptr_out0]] \n"                    /* dup v10, bias */
                    // "ldp    q10, q11, [%[rst_mask]] \n"                    /* dup v10, bias */

                    //r1
                    "smlal  v17.8h,  %[v4].8b,  v2.8b   \n"       /* outr00 = 02468 * w01 */
                    "smlal  v18.8h,  %[v5].8b,  v3.8b\n"         /* outr00 += 13579 * w02 */
                    "smlal  v19.8h,  %[v3].8b,  v7.8b\n"         /* outr00 += 013579 * w00 */

                    "saddw   v12.4s, v12.4s, v17.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v17.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v18.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v18.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v19.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"             /* v11 += outr00.high*/

                    //r2
                    "smull  v17.8h,  %[v7].8b,  v4.8b   \n"       /* outr00 = 02468 * w01 */
                    "smull  v18.8h,  %[v8].8b,  v5.8b\n"         /* outr00 += 13579 * w02 */
                    "smull  v19.8h,  %[v6].8b,  v8.8b\n"         /* outr00 += 013579 * w00 */

                    "saddw   v12.4s, v12.4s, v17.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v17.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v18.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v18.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v19.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"             /* v11 += outr00.high*/

                    // "bif v12.16b, v0.16b, v10.16b         \n"
                    // "bif v13.16b, v1.16b, v11.16b         \n"

                    "stp     q12, q13, [%[ptr_out0]]   \n"      /* store q10, q11 -> ptr_out   */
                    :[din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [vmask] "+r" (val_mask)
                    : [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), [bias_val] "r" (vbias), \
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22), \
                        [rst_mask] "r" (rmask), [ptr_out0] "r"(out_buf1)
                    :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", \
                      "v10", "v11", "v12","v13","v14","v15", "v16", "v17", "v18", "v19", "v20"
                );
#else
                 unsigned int* rst_mask = rmask;
                //prefetch input
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
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6
                    "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vmov.u32 d11, #0                   @ zero\n"

                    "vdup.s8     d5, d0[3]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d6, d0[4]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d7, d0[5]               @ d4 = w02, w02, w02, w02\n"

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                    "vext.8  d18, d11, d13, #7     @ ext \n" //d16 = -1 1 3 5
                    "vext.8  d19, d11, d15, #7     @ ext \n" //d17 = -1 1 3 5
                    "vext.8  d20, d11, d17, #7     @ ext \n" //d18 = -1 1 3 5

                    // "pld [%[dout_ptr1]]                @ preload data\n"

                    //r0
                    "vmull.s8 q13, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vmull.s8 q14, d13, d4                 @ out1 = din0 * w02 \n" // q12 = d12 * w02
                    "vmull.s8 q15, d18, d2                 @ out2 = din0 * w00 \n" // q12 = d12 * w02

                    "vdup.s8 d8, d0[6]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8 d9, d0[7]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8 d10, d1[0]               @ d4 = w02, w02, w02, w02\n"

                    //out0
                    "vdup.32 q11, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                            @ and \n" //q9 = vbias

                    //r1
                    "vmlal.s8 q13, d14, d6                 @ out0 += din1 * w11 \n" // q12 = d12 * w11
                    "vmlal.s8 q14, d15, d7                 @ out1 += din1 * w12 \n" // q12 = d12 * w11
                    "vmlal.s8 q15, d19, d5                 @ out2 += din1 * w10 \n" // q12 = d12 * w11

                    // "vld1.32 {d12-d13}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    // "vld1.32 {d14-d15}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    //r2
                    "vmull.s8 q13, d16, d9                 @ out0 += din1 * w21 \n" // q12 = d12 * w11
                    "vmull.s8 q14, d17, d10                 @ out1 += din1 * w22 \n" // q12 = d12 * w11
                    "vmull.s8 q15, d20, d8                 @ out2 += din1 * w20 \n" // q12 = d12 * w11

                    // "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    // "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    // "sub %[dout_ptr1], #16                  @ sub \n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    // "vbif q11, q6, q1        @ bit select, deal with right pad\n"
                    // "vbif q12, q7, q2       @ bit select, deal with right pad\n"

                    "vst1.32 {d22-d25}, [%[dout_ptr1]]         @ store\n"
                    // "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask), [size_pad_right] "r" (size_pad_right), [dout_ptr1] "r"(out_buf1)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"
                );
#endif
                for (int w = 0; w < w_out; ++w){
                    *doutr0++ = out_buf1[w];
                }
                dout_ptr += w_out;
            }
        }
    }
    // write_out
    int32_to_dtype<Dtype>(pre_out, dout, scale, ch_in, num, h_out * w_out);
}

//1 line w_in > 16
template <typename Dtype>
void conv_depthwise_3x3s2p1_bias_relu_int8(Dtype* dout,
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
    //! for 4x6 convolution window
    const uint8_t right_pad_idx[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    int8_t* zero_ptr = ctx->workspace_data<int8_t>();
    memset(zero_ptr, 0, w_in * sizeof(int8_t));
    int* write_ptr = reinterpret_cast<int*>(zero_ptr) + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = (w_in + 15) >> 4;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(w_in - 15 - (cnt_col << 4));
    if (size_pad_right == 17){
        size_pad_right = 0;
        cnt_col++;
    }

    uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned int rst_remain = (unsigned int)(w_out - ((cnt_col + 1) << 3));
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    int8x8_t vzero = vdup_n_s8(0);
    int32x4_t vzero_32 = vdupq_n_s32(0);

    uint8x16_t vmask_rp = vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
    uint8_t vmask[16];
    vst1q_u8(vmask, vmask_rp);

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);
    int32_t* pre_out = reinterpret_cast<int*>(write_ptr + w_out + 4);
    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = pre_out + n * ch_in * size_out_channel;

#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? static_cast<int>(bias[c] / scale[c]) : 0;

            const signed char* wei_ptr = weights + c * w_stride;
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

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;

                doutr0 = dout_ptr;
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr1 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr0 + w_in;
                    dr2 = dr1 + w_in;
                }
                //! process bottom pad
                if (i + 2 > h_in) {
                    switch (i + 2 - h_in) {
                        case 2:
                            din_ptr1 = zero_ptr;
                        case 1:
                            din_ptr2 = zero_ptr;
                        default:
                            break;
                    }
                }
                int cnt = cnt_col;
#ifdef __aarch64__
                uint8_t *val_mask = vmask;
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "movi   v10.4s, #0x0\n"
                // left
                    "ld2    {v0.8b - v1.8b}, [%[din_ptr0]]         \n"         /*load a00-a015 to q0*/
                    "ld2    {v2.8b - v3.8b}, [%[din_ptr1]]         \n"        /* load a00-a015 to q0*/
                    "ld2    {v4.8b - v5.8b}, [%[din_ptr2]]         \n"         /*load a00-a015 to q0*/

                    "ld1    {v12.4s}, [%[bias_val]] \n"                    /* dup v10, bias*/
                    "ld1    {v13.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    "ext v6.8b, v10.8b, v1.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 013579 */
                    "ext v7.8b, v10.8b, v3.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 013579 */
                    "ext v8.8b, v10.8b, v5.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 013579 */

                    //r0
                    "smull  v14.8h,  %[v1].8b,  v0.8b   \n"       /* outr00 = 02468 * w01 */
                    "smull  v15.8h,  %[v2].8b,  v1.8b\n"         /* outr00 += 13579 * w02 */
                    "smull  v16.8h,  %[v0].8b,  v6.8b\n"         /* outr00 += 013579 * w00 */

                    "add   %[din_ptr0], %[din_ptr0], #15                       \n"
                    "add   %[din_ptr1], %[din_ptr1], #15                       \n"
                    "add   %[din_ptr2], %[din_ptr2], #15                       \n"

                    //r1
                    "smlal  v14.8h,  %[v4].8b,  v2.8b   \n"       /* outr00 = 02468 * w01 */
                    "smlal  v15.8h,  %[v5].8b,  v3.8b\n"         /* outr00 += 13579 * w02 */
                    "smlal  v16.8h,  %[v3].8b,  v7.8b\n"         /* outr00 += 013579 * w00 */

                    "saddw   v12.4s, v12.4s, v14.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v14.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v15.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v15.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v16.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v16.8h        \n"             /* v11 += outr00.high*/

                    //r2
                    "smull  v14.8h,  %[v7].8b,  v4.8b   \n"       /* outr00 = 02468 * w01 */
                    "smull  v15.8h,  %[v8].8b,  v5.8b\n"         /* outr00 += 13579 * w02 */
                    "smull  v16.8h,  %[v6].8b,  v8.8b\n"         /* outr00 += 013579 * w00 */

                    "ld2    {v0.8b - v1.8b}, [%[din_ptr0]], #16         \n"         /*load a00-a015 to q0*/
                    "ld2    {v2.8b - v3.8b}, [%[din_ptr1]], #16         \n"        /* load a00-a015 to q0*/
                    "ld2    {v4.8b - v5.8b}, [%[din_ptr2]], #16         \n"         /*load a00-a015 to q0*/

                    "saddw   v12.4s, v12.4s, v14.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v14.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v15.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v15.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v16.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v16.8h        \n"             /* v11 += outr00.high*/

                    "smax   v12.4s, v12.4s, v10.4s    \n"         /*relu*/
                    "smax   v13.4s, v13.4s, v10.4s    \n"         /*relu*/

                    "stp     q12, q13, [%[ptr_out0]], #32   \n"      /* store q10, q11 -> ptr_out   */

                    "ld1    {v12.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v13.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    "cmp  %[cnt], #1                \n"
                    "blt 3f                         \n"
                //mid
                    "1:                             \n"
                    "ld1    {v6.8b}, [%[din_ptr0]]         \n"         /*load a00-a015 to q0*/
                    "ld1    {v7.8b}, [%[din_ptr1]]         \n"         /*load a00-a015 to q0*/
                    "ld1    {v8.8b}, [%[din_ptr2]]         \n"         /*load a00-a015 to q0*/

                    "ext v9.8b, v0.8b, v6.8B, #1       \n" /* vext_s8(vzero, vinr0, 7); 246810 */
                    "ext v11.8b, v2.8b, v7.8B, #1       \n" /* vext_s8(vzero, vinr0, 7); 246810 */
                    "ext v14.8b, v4.8b, v8.8B, #1       \n" /* vext_s8(vzero, vinr0, 7); 246810 */

                     //r0
                    "smull  v6.8h,  %[v0].8b,  v0.8b   \n"       /* outr00 = 02468 * w00 */
                    "smull  v7.8h,  %[v1].8b,  v1.8b\n"         /* outr00 += 13579 * w01 */
                    "smull  v8.8h,  %[v2].8b,  v9.8b\n"         /* outr00 += 246810 * w02 */

                    //r1
                    "smlal  v6.8h,  %[v3].8b,  v2.8b   \n"       /* outr00 = 02468 * w00 */
                    "smlal  v7.8h,  %[v4].8b,  v3.8b\n"         /* outr00 += 13579 * w01 */
                    "smlal  v8.8h,  %[v5].8b,  v11.8b\n"         /* outr00 += 246810 * w02 */

                    "saddw   v12.4s, v12.4s, v6.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v6.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v7.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v7.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v8.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v8.8h        \n"             /* v11 += outr00.high*/

                    //r2
                    "smull  v6.8h,  %[v6].8b,  v4.8b   \n"       /* outr00 = 02468 * w00 */
                    "smull  v7.8h,  %[v7].8b,  v5.8b\n"         /* outr00 += 13579 * w01 */
                    "smull  v8.8h,  %[v8].8b,  v14.8b\n"         /* outr00 += 246810 * w02 */

                    "ld2    {v0.8b - v1.8b}, [%[din_ptr0]], #16         \n"         /*load a00-a015 to q0*/
                    "ld2    {v2.8b - v3.8b}, [%[din_ptr1]], #16         \n"        /* load a00-a015 to q0*/
                    "ld2    {v4.8b - v5.8b}, [%[din_ptr2]], #16         \n"         /*load a00-a015 to q0*/

                    "saddw   v12.4s, v12.4s, v6.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v6.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v7.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v7.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v8.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v8.8h        \n"             /* v11 += outr00.high*/

                    "smax   v12.4s, v12.4s, v10.4s    \n"         /*relu*/
                    "smax   v13.4s, v13.4s, v10.4s    \n"         /*relu*/

                    "subs %[cnt], %[cnt], #1               \n"

                    "stp     q12, q13, [%[ptr_out0]], #32   \n"      /* store q10, q11 -> ptr_out   */

                    "ld1    {v12.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "ld1    {v13.4s}, [%[bias_val]] \n"                    /* dup v10, bias */
                    "bne 1b                         \n"
                //right
                    "3:                             \n"
                    "ld1 {v14.8b}, [%[vmask]], #8             \n"
                    "ld1 {v15.8b}, [%[vmask]]                \n"

                    "bif v0.8b, v10.8b, v14.8b               \n"
                    "bif v1.8b, v10.8b, v15.8b               \n"
                    "bif v2.8b, v10.8b, v14.8b               \n"
                    "bif v3.8b, v10.8b, v15.8b               \n"
                    "bif v4.8b, v10.8b, v14.8b               \n"
                    "bif v5.8b, v10.8b, v15.8b               \n"

                    "ext v6.8b, v0.8b, v10.8B, #1       \n" /* vext_s8(vzero, vinr0, 7); 2468.. */
                    "ext v7.8b, v2.8b, v10.8B, #1       \n" /* vext_s8(vzero, vinr0, 7); 2468..*/
                    "ext v8.8b, v4.8b, v10.8B, #1       \n" /* vext_s8(vzero, vinr0, 7); 2468.. */

                    //r0
                    "smull  v14.8h,  %[v0].8b,  v0.8b   \n"       /* outr00 = 02468 * w00 */
                    "smull  v15.8h,  %[v1].8b,  v1.8b\n"         /* outr00 += 13579 * w01 */
                    "smull  v16.8h,  %[v2].8b,  v6.8b\n"         /* outr00 += 246810 * w02 */

                    //r1
                    "smlal  v14.8h,  %[v3].8b,  v2.8b   \n"       /* outr00 = 02468 * w00 */
                    "smlal  v15.8h,  %[v4].8b,  v3.8b\n"         /* outr00 += 13579 * w01 */
                    "smlal  v16.8h,  %[v5].8b,  v7.8b\n"         /* outr00 += 246810 * w02 */

                    "saddw   v12.4s, v12.4s, v14.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v14.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v15.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v15.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v16.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v16.8h        \n"             /* v11 += outr00.high*/

                    //r2
                    "smull  v14.8h,  %[v6].8b,  v4.8b   \n"       /* outr00 = 02468 * w00 */
                    "smull  v15.8h,  %[v7].8b,  v5.8b\n"         /* outr00 += 13579 * w01 */
                    "smull  v16.8h,  %[v8].8b,  v8.8b\n"         /* outr00 += 246810 * w02 */

                    "ldp    q0, q1, [%[ptr_out0]] \n"                    /* dup v10, bias */
                    "ldp    q9, q11, [%[rst_mask]] \n"                    /* dup v10, bias */

                    "saddw   v12.4s, v12.4s, v14.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v14.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v15.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v15.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v16.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v16.8h        \n"             /* v11 += outr00.high*/

                    "smax   v12.4s, v12.4s, v10.4s    \n"         /*relu*/
                    "smax   v13.4s, v13.4s, v10.4s    \n"         /*relu*/

                    "bif v12.16b, v0.16b, v9.16b         \n"
                    "bif v13.16b, v1.16b, v11.16b         \n"

                    "stp     q12, q13, [%[ptr_out0]], #32 \n"      /* store q10, q11 -> ptr_out       */

                    :[cnt]"+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [ptr_out0] "+r"(doutr0), [vmask] "+r" (val_mask)
                    : [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), [bias_val] "r" (vbias), \
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22), [rst_mask] "r" (rmask)
                    :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", \
                      "v10", "v11", "v12","v13","v14","v15", "v16"
                );
#else
                unsigned int* rst_mask = rmask;
                //prefetch input
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
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6
                    "vmov.u32 d11, #0                   @ zero\n"

                    "vdup.s8     d5, d0[3]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d6, d0[4]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d7, d0[5]               @ d4 = w02, w02, w02, w02\n"

                    "vext.8  d18, d11, d13, #7     @ ext \n" //d16 = -1 1 3 5
                    "vext.8  d19, d11, d15, #7     @ ext \n" //d17 = -1 1 3 5
                    "vext.8  d20, d11, d17, #7     @ ext \n" //d18 = -1 1 3 5

                    //r0
                    "vmull.s8 q13, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vmull.s8 q14, d13, d4                 @ out1 = din0 * w02 \n" // q12 = d12 * w02
                    "vmull.s8 q15, d18, d2                 @ out2 = din0 * w00 \n" // q12 = d12 * w02

                    "vdup.s8 d8, d0[6]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8 d9, d0[7]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8 d10, d1[0]               @ d4 = w02, w02, w02, w02\n"

                    //out0
                    "vdup.32 q11, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                            @ and \n" //q9 = vbias

                    //r1
                    "vmlal.s8 q13, d14, d6                 @ out0 += din1 * w11 \n" // q12 = d12 * w11
                    "vmlal.s8 q14, d15, d7                 @ out1 += din1 * w12 \n" // q12 = d12 * w11
                    "vmlal.s8 q15, d19, d5                 @ out2 += din1 * w10 \n" // q12 = d12 * w11

                    "add %[din_ptr0], #15                   @add \n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "add %[din_ptr1], #15                   @add \n"

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "add %[din_ptr2], #15                   @add \n"

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    //r2
                    "vmull.s8 q13, d16, d9                 @ out0 += din1 * w21 \n" // q12 = d12 * w11
                    "vmull.s8 q14, d17, d10                 @ out1 += din1 * w22 \n" // q12 = d12 * w11
                    "vmull.s8 q15, d20, d8                 @ out2 += din1 * w20 \n" // q12 = d12 * w11

                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmov.u32 q8, #0                        @ max \n" //max

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmax.s32 q11, q11, q8                      @ max\n"
                    "vmax.s32 q12, q12, q8                      @ max\n"

                    "vst1.32 {d22-d23}, [%[dout_ptr1]]!         @ store\n"
                    "cmp %[cnt], #1                                 \n"
                    "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    "blt 1f                                         \n"

                //mid
                    "2:                                              \n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]!    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]!    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]!    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6

                    "vld1.8 {d21}, [%[din_ptr0]]    @ load din00= 16 17\n" //d10 = 0 2 4 6
                    "vld1.8 {d22}, [%[din_ptr1]]    @ load din00= 16 17\n" //d12 = 0 2 4 6
                    "vld1.8 {d23}, [%[din_ptr2]]    @ load din00= 16 17\n" //d14 = 0 2 4 6

                    "vext.8  d18, d12, d21, #1     @ ext din00 = 2 4 6 8\n" //d16 = 2 4 6 8
                    "vext.8  d19, d14, d22, #1     @ ext \n" //d17 = 2 4 6 8
                    "vext.8  d20, d16, d23, #1     @ ext \n" //d18 = 2 4 6 8

                    //r0
                    "vmull.s8 q13, d12, d2                 @ out0 = din0 * w00 \n" // q12 = 0 2 4 6
                    "vmull.s8 q14, d13, d3                 @ out1 = din0 * w01 \n" // q12 = 1 3 5 7
                    "vmull.s8 q15, d18, d4                 @ out2 = din0 * w02 \n" // q12 = 2 4 6 8

                    //out0
                    "vdup.32 q11, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                            @ and \n" //q9 = vbias

                    //r1
                    "vmlal.s8 q13, d14, d5                 @ out0 += din1 * w10 \n" // q12 = 0 2 4 6
                    "vmlal.s8 q14, d15, d6                 @ out1 += din1 * w11 \n" // q12 = 1 3 5 7
                    "vmlal.s8 q15, d19, d7                 @ out2 += din1 * w12 \n" // q12 = 2 4 6 8

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    //r2
                    "vmull.s8 q13, d16, d8                 @ out0 += din1 * w20 \n" // q12 = 0 2 4 6
                    "vmull.s8 q14, d17, d9                 @ out1 += din1 * w21 \n" // q12 = 1 3 5 7
                    "vmull.s8 q15, d20, d10                 @ out2 += din1 * w22 \n" // q12 = 2 4 6 8

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vmov.u32 q8, #0                          @ mov \n"

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"

                    "vmax.s32 q11, q11, q8                      @ max\n"
                    "vmax.s32 q12, q12, q8                      @ max\n"

                    "vst1.32 {d22-d23}, [%[dout_ptr1]]!         @ store\n"

                    "subs %[cnt], #1                                \n"
                    "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    "bne  2b                                        \n"
                //right
                    "1:                                              \n"
                    "cmp %[size_pad_right], #1                       \n"
                    "blt 3f                                         \n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]!    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]!    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]!    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6
                    "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    //out0
                    "vdup.32 q11, %[bias]                 @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                 @ and \n" //q9 = vbias

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                    "vext.8  d18, d12, d11, #1     @ ext din00 = 2 4 6 8\n" //d16 = -1 1 3 5
                    "vext.8  d19, d14, d11, #1     @ ext \n" //d17 = -1 1 3 5
                    "vext.8  d20, d16, d11, #1     @ ext \n" //d18 = -1 1 3 5

                    //r0
                    "vmull.s8 q13, d12, d2                 @ out0 = din0 * w00 \n" // q12 = 0 2 4 6
                    "vmull.s8 q14, d13, d3                 @ out1 = din0 * w01 \n" // q12 = 1 3 5 7
                    "vmull.s8 q15, d18, d4                 @ out2 = din0 * w02 \n" // q12 = 2 4 6 8

                    //r1
                    "vmlal.s8 q13, d14, d5                 @ out0 += din1 * w11 \n" // q12 = 0 2 4 6
                    "vmlal.s8 q14, d15, d6                 @ out1 += din1 * w12 \n" // q12 = 1 3 5 7
                    "vmlal.s8 q15, d19, d7                 @ out2 += din1 * w10 \n" // q12 = 2 4 6 8

                    "vld1.32 {d12-d13}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d14-d15}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    //r2
                    "vmull.s8 q13, d16, d8                 @ out0 += din1 * w11 \n" // q12 = 0 2 4 6
                    "vmull.s8 q14, d17, d9                 @ out1 += din1 * w12 \n" // q12 = 1 3 5 7
                    "vmull.s8 q15, d20, d10                 @ out2 += din1 * w10 \n" // q12 = 2 4 6 8

                    "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "sub %[dout_ptr1], #16                  @ sub \n"
                    "vmov.u32 q8, #0                         @mov \n"
                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmax.s32 q11, q11, q8                      @ max\n"
                    "vmax.s32 q12, q12, q8                      @ max\n"

                    "vbif q11, q6, q1        @ bit select, deal with right pad\n"
                    "vbif q12, q7, q2       @ bit select, deal with right pad\n"

                    "vst1.32 {d22-d23}, [%[dout_ptr1]]!         @ store\n"
                    "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    "3:                                             \n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [dout_ptr1] "+r"(doutr0), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask), [size_pad_right] "r" (size_pad_right)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"
                );
#endif
                dout_ptr += w_out;
            }
        }
    }
    // write_out
    int32_to_dtype<Dtype>(pre_out, dout, scale, ch_in, num, h_out * w_out);
}

template <typename Dtype>
void conv_depthwise_3x3s2p1_bias_s_relu_int8(Dtype* dout,
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
    // const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const uint8_t right_pad_idx[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

   int8_t* zero_ptr = ctx->workspace_data<int8_t>();
    memset(zero_ptr, 0, w_in * sizeof(int8_t));
    int* write_ptr = reinterpret_cast<int*>(zero_ptr) + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    unsigned int size_pad_right = (unsigned int)(w_in);

    uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned int rst_remain = (unsigned int)w_out;
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    uint8x16_t vmask_rp = vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
    uint8_t vmask[16];
    vst1q_u8(vmask, vmask_rp);

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);
    int8x8_t vzero = vdup_n_s8(0);
    int32x4_t vzero_32 = vdupq_n_s32(0);
    int32_t* pre_out = reinterpret_cast<int*>(write_ptr + w_out + 4);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = pre_out + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? static_cast<int>(bias[c] / scale[c]) : 0;

            const signed char* wei_ptr = weights + c * w_stride;

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

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;

                doutr0 = dout_ptr;

                int out_buf1[8];
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr1 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr2 + w_in;
                    dr2 = dr1 + w_in;
                }
                //! process bottom pad
                if (i + 2 > h_in) {
                    switch (i + 2 - h_in) {
                        case 2:
                            din_ptr1 = zero_ptr;
                        case 1:
                            din_ptr2 = zero_ptr;
                        default:
                            break;
                    }
                }
#ifdef __aarch64__
                unsigned int* rst_mask = rmask;
                uint8_t* val_mask = vmask;
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "movi   v16.4s, #0x0\n"
                // left
                    "ld1 {v10.8b}, [%[vmask]], #8             \n"
                    "ld1 {v11.8b}, [%[vmask]]                \n"
                    "ld2    {v0.8b - v1.8b}, [%[din_ptr0]]         \n"         /*load a00-a015 to q0*/
                    "ld2    {v2.8b - v3.8b}, [%[din_ptr1]]         \n"        /* load a00-a015 to q0*/
                    "ld2    {v4.8b - v5.8b}, [%[din_ptr2]]         \n"         /*load a00-a015 to q0*/

                    "bif v0.8b, v16.8b, v10.8b               \n"
                    "bif v1.8b, v16.8b, v11.8b               \n"
                    "bif v2.8b, v16.8b, v10.8b               \n"
                    "bif v3.8b, v16.8b, v11.8b               \n"
                    "bif v4.8b, v16.8b, v10.8b               \n"
                    "bif v5.8b, v16.8b, v11.8b               \n"

                    "ld1    {v12.4s}, [%[bias_val]] \n"                    /* dup v10, bias*/
                    "ld1    {v13.4s}, [%[bias_val]] \n"                    /* dup v10, bias */

                    "ext v6.8b, v16.8b, v1.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 013579 */
                    "ext v7.8b, v16.8b, v3.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 013579 */
                    "ext v8.8b, v16.8b, v5.8B, #7       \n" /* vext_s8(vzero, vinr0, 7); 013579 */

                    //r0
                    "smull  v17.8h,  %[v1].8b,  v0.8b   \n"       /* outr00 = 02468 * w01 */
                    "smull  v18.8h,  %[v2].8b,  v1.8b\n"         /* outr00 += 13579 * w02 */
                    "smull  v19.8h,  %[v0].8b,  v6.8b\n"         /* outr00 += 013579 * w00 */

                    // "ldp    q0, q1, [%[ptr_out0]] \n"                    /* dup v10, bias */
                    // "ldp    q10, q11, [%[rst_mask]] \n"                    /* dup v10, bias */

                    //r1
                    "smlal  v17.8h,  %[v4].8b,  v2.8b   \n"       /* outr00 = 02468 * w01 */
                    "smlal  v18.8h,  %[v5].8b,  v3.8b\n"         /* outr00 += 13579 * w02 */
                    "smlal  v19.8h,  %[v3].8b,  v7.8b\n"         /* outr00 += 013579 * w00 */

                    "saddw   v12.4s, v12.4s, v17.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v17.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v18.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v18.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v19.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"             /* v11 += outr00.high*/

                    //r2
                    "smull  v17.8h,  %[v7].8b,  v4.8b   \n"       /* outr00 = 02468 * w01 */
                    "smull  v18.8h,  %[v8].8b,  v5.8b\n"         /* outr00 += 13579 * w02 */
                    "smull  v19.8h,  %[v6].8b,  v8.8b\n"         /* outr00 += 013579 * w00 */

                    "saddw   v12.4s, v12.4s, v17.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v17.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v18.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v18.8h        \n"             /* v11 += outr00.high*/

                    "saddw   v12.4s, v12.4s, v19.4h         \n"             /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"             /* v11 += outr00.high*/

                    "smax   v12.4s, v12.4s, v16.4s    \n"         /*relu*/
                    "smax   v13.4s, v13.4s, v16.4s    \n"         /*relu*/

                    // "bif v12.16b, v0.16b, v10.16b         \n"
                    // "bif v13.16b, v1.16b, v11.16b         \n"

                    "stp     q12, q13, [%[ptr_out0]]   \n"      /* store q10, q11 -> ptr_out   */
                    :[din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                         [vmask] "+r" (val_mask)
                    : [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), [bias_val] "r" (vbias), \
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22), \
                        [rst_mask] "r" (rmask), [ptr_out0] "r"(out_buf1)
                    :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", \
                      "v10", "v11", "v12","v13","v14","v15", "v16", "v17", "v18", "v19", "v20"
                );

#else
                unsigned int* rst_mask = rmask;
                //prefetch input
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
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6
                    "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vmov.u32 d11, #0                   @ zero\n"

                    "vdup.s8     d5, d0[3]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d6, d0[4]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d7, d0[5]               @ d4 = w02, w02, w02, w02\n"

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                    "vext.8  d18, d11, d13, #7     @ ext \n" //d16 = -1 1 3 5
                    "vext.8  d19, d11, d15, #7     @ ext \n" //d17 = -1 1 3 5
                    "vext.8  d20, d11, d17, #7     @ ext \n" //d18 = -1 1 3 5

                    // "pld [%[dout_ptr1]]                @ preload data\n"

                    //r0
                    "vmull.s8 q13, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vmull.s8 q14, d13, d4                 @ out1 = din0 * w02 \n" // q12 = d12 * w02
                    "vmull.s8 q15, d18, d2                 @ out2 = din0 * w00 \n" // q12 = d12 * w02

                    "vdup.s8 d8, d0[6]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8 d9, d0[7]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8 d10, d1[0]               @ d4 = w02, w02, w02, w02\n"

                    //out0
                    "vdup.32 q11, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                            @ and \n" //q9 = vbias

                    //r1
                    "vmlal.s8 q13, d14, d6                 @ out0 += din1 * w11 \n" // q12 = d12 * w11
                    "vmlal.s8 q14, d15, d7                 @ out1 += din1 * w12 \n" // q12 = d12 * w11
                    "vmlal.s8 q15, d19, d5                 @ out2 += din1 * w10 \n" // q12 = d12 * w11

                    // "vld1.32 {d12-d13}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    // "vld1.32 {d14-d15}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    //r2
                    "vmull.s8 q13, d16, d9                 @ out0 += din1 * w21 \n" // q12 = d12 * w11
                    "vmull.s8 q14, d17, d10                 @ out1 += din1 * w22 \n" // q12 = d12 * w11
                    "vmull.s8 q15, d20, d8                 @ out2 += din1 * w20 \n" // q12 = d12 * w11

                    // "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    // "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    // "sub %[dout_ptr1], #16                  @ sub \n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vmov.u32 q8, #0                         @ mov \n"

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmax.s32 q11, q11, q8                      @ max\n"
                    "vmax.s32 q12, q12, q8                      @ max\n"

                    // "vbif q11, q6, q1        @ bit select, deal with right pad\n"
                    // "vbif q12, q7, q2       @ bit select, deal with right pad\n"

                    "vst1.32 {d22-d25}, [%[dout_ptr1]]         @ store\n"
                    // "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask), [size_pad_right] "r" (size_pad_right), [dout_ptr1] "r"(out_buf1)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"
                );
#endif
                for (int w = 0; w < w_out; ++w){
                    *doutr0++ = out_buf1[w];
                }
                dout_ptr += w_out;
            }
        }
    }
    // write_out
    int32_to_dtype<Dtype>(pre_out, dout, scale, ch_in, num, h_out * w_out);
}

template void conv_depthwise_3x3s2p1_int8<int8_t>(int8_t* dout,
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

template void conv_depthwise_3x3s2p1_int8<float>(float* dout,
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

template void conv_depthwise_3x3s2p1_bias_int8<int8_t>(int8_t* dout,
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

template void conv_depthwise_3x3s2p1_bias_int8<float>(float* dout,
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

template void conv_depthwise_3x3s2p1_bias_s_int8<int8_t>(int8_t* dout,
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

template void conv_depthwise_3x3s2p1_bias_s_int8<float>(float* dout,
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

template void conv_depthwise_3x3s2p1_bias_relu_int8<int8_t>(int8_t* dout,
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

template void conv_depthwise_3x3s2p1_bias_relu_int8<float>(float* dout,
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

template void conv_depthwise_3x3s2p1_bias_s_relu_int8<int8_t>(int8_t* dout,
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

template void conv_depthwise_3x3s2p1_bias_s_relu_int8<float>(float* dout,
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

