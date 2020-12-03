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

#include "lite/kernels/arm/layout_compute.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

<<<<<<< HEAD
#define NCHWTONHWC(type)                                                  \
  auto& param = this->template Param<param_t>();                          \
  auto input = param.x->template data<type>();                            \
  auto input_dim = param.x->dims();                                       \
  CHECK(input_dim.size() == 4)                                            \
      << "NCHW to NHWC should guarantee that the input dims should be 4"; \
  int n = input_dim[0];                                                   \
  int c = input_dim[1];                                                   \
  int h = input_dim[2];                                                   \
  int w = input_dim[3];                                                   \
  param.y->Resize({n, h, w, c});                                          \
  auto output = param.y->template mutable_data<type>(TARGET(kARM));       \
  if (c == 1) {                                                           \
    memcpy(output, input, sizeof(type) * n * h * w);                      \
    return;                                                               \
  }                                                                       \
  lite::arm::math::NCHW2NHWC<type>(n, c, h * w, input, output);

#define NHWCTONCHW(type)                                                  \
  auto& param = this->template Param<param_t>();                          \
  auto input = param.x->template data<type>();                            \
  auto input_dim = param.x->dims();                                       \
  CHECK(input_dim.size() == 4)                                            \
      << "NHWC to NCHW should guarantee that the input dims should be 4"; \
  int n = input_dim[0];                                                   \
  int h = input_dim[1];                                                   \
  int w = input_dim[2];                                                   \
  int c = input_dim[3];                                                   \
  param.y->Resize({n, c, h, w});                                          \
  auto output = param.y->template mutable_data<type>(TARGET(kARM));       \
  if (c == 1) {                                                           \
    memcpy(output, input, sizeof(type) * n * h * w);                      \
    return;                                                               \
  }                                                                       \
  lite::arm::math::NHWC2NCHW<type>(n, c, h * w, input, output);

void NCHWToNHWCCompute::Run() { NCHWTONHWC(float) }

void NCHWToNHWCComputeInt8::Run() { NCHWTONHWC(int8_t) }

void NHWCToNCHWCompute::Run() { NHWCTONCHW(float) }

void NHWCToNCHWComputeInt8::Run() { NHWCTONCHW(int8_t) }
=======
#define NCHWTONHWC(type)                                                 \
  auto& param = this->template Param<param_t>();                         \
  auto input = param.x->template data<type>();                           \
  auto input_dim = param.x->dims();                                      \
  if (input_dim.size() != 4) {                                           \
    LOG(WARNING) << "NCHW to NHWC should guarantee that the input dims " \
                    "should be 4, but received "                         \
                 << input_dim.size();                                    \
    param.y->ShareDataWith(*param.x);                                    \
    return;                                                              \
  }                                                                      \
  int n = input_dim[0];                                                  \
  int c = input_dim[1];                                                  \
  int h = input_dim[2];                                                  \
  int w = input_dim[3];                                                  \
  param.y->Resize({n, h, w, c});                                         \
  auto output = param.y->template mutable_data<type>(TARGET(kARM));      \
  if (c == 1) {                                                          \
    memcpy(output, input, sizeof(type) * n * h * w);                     \
    return;                                                              \
  }                                                                      \
  lite::arm::math::NCHW2NHWC<type>(n, c, h * w, input, output);

#define NHWCTONCHW(type)                                                 \
  auto& param = this->template Param<param_t>();                         \
  auto input = param.x->template data<type>();                           \
  auto input_dim = param.x->dims();                                      \
  if (input_dim.size() != 4) {                                           \
    LOG(WARNING) << "NHWC to NCHW should guarantee that the input dims " \
                    "should be 4, but received "                         \
                 << input_dim.size();                                    \
    param.y->ShareDataWith(*param.x);                                    \
    return;                                                              \
  }                                                                      \
  int n = input_dim[0];                                                  \
  int h = input_dim[1];                                                  \
  int w = input_dim[2];                                                  \
  int c = input_dim[3];                                                  \
  param.y->Resize({n, c, h, w});                                         \
  auto output = param.y->template mutable_data<type>(TARGET(kARM));      \
  if (c == 1) {                                                          \
    memcpy(output, input, sizeof(type) * n * h * w);                     \
    return;                                                              \
  }                                                                      \
  lite::arm::math::NHWC2NCHW<type>(n, c, h * w, input, output);

template <>
void NCHWToNHWCCompute<PRECISION(kFloat)>::Run() {
  NCHWTONHWC(float);
}

template <>
void NCHWToNHWCCompute<PRECISION(kInt8)>::Run() {
  NCHWTONHWC(int8_t);
}

template <>
void NHWCToNCHWCompute<PRECISION(kFloat)>::Run() {
  NHWCTONCHW(float);
}

template <>
void NHWCToNCHWCompute<PRECISION(kInt8)>::Run() {
  NHWCTONCHW(int8_t);
}
>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

<<<<<<< HEAD
REGISTER_LITE_KERNEL(layout,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::NCHWToNHWCCompute,
                     nchw2nhwc)
=======
typedef paddle::lite::kernels::arm::NCHWToNHWCCompute<PRECISION(kFloat)>
    NCHW_fp32;
typedef paddle::lite::kernels::arm::NCHWToNHWCCompute<PRECISION(kInt8)>
    NCHW_int8;
typedef paddle::lite::kernels::arm::NHWCToNCHWCompute<PRECISION(kFloat)>
    NHWC_fp32;
typedef paddle::lite::kernels::arm::NHWCToNCHWCompute<PRECISION(kInt8)>
    NHWC_int8;

REGISTER_LITE_KERNEL(layout, kARM, kFloat, kNCHW, NCHW_fp32, nchw2nhwc)
>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

<<<<<<< HEAD
REGISTER_LITE_KERNEL(layout,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::NHWCToNCHWCompute,
                     nhwc2nchw)
=======
REGISTER_LITE_KERNEL(layout, kARM, kFloat, kNCHW, NHWC_fp32, nhwc2nchw)
>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

<<<<<<< HEAD
REGISTER_LITE_KERNEL(layout,
                     kARM,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::arm::NCHWToNHWCComputeInt8,
                     int8_nchw2nhwc)
=======
REGISTER_LITE_KERNEL(layout, kARM, kInt8, kNCHW, NCHW_int8, int8_nchw2nhwc)
>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

<<<<<<< HEAD
REGISTER_LITE_KERNEL(layout,
                     kARM,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::arm::NHWCToNCHWComputeInt8,
                     int8_nhwc2nchw)
=======
REGISTER_LITE_KERNEL(layout, kARM, kInt8, kNCHW, NHWC_int8, int8_nhwc2nchw)
>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

<<<<<<< HEAD
REGISTER_LITE_KERNEL(layout_once,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::NCHWToNHWCCompute,
                     nchw2nhwc)
=======
REGISTER_LITE_KERNEL(layout_once, kARM, kFloat, kNCHW, NCHW_fp32, nchw2nhwc)
>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

<<<<<<< HEAD
REGISTER_LITE_KERNEL(layout_once,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::NHWCToNCHWCompute,
                     nhwc2nchw)
=======
REGISTER_LITE_KERNEL(layout_once, kARM, kFloat, kNCHW, NHWC_fp32, nhwc2nchw)
>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

<<<<<<< HEAD
REGISTER_LITE_KERNEL(layout_once,
                     kARM,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::arm::NCHWToNHWCComputeInt8,
                     int8_nchw2nhwc)
=======
REGISTER_LITE_KERNEL(layout_once, kARM, kInt8, kNCHW, NCHW_int8, int8_nchw2nhwc)
>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

<<<<<<< HEAD
REGISTER_LITE_KERNEL(layout_once,
                     kARM,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::arm::NHWCToNCHWComputeInt8,
                     int8_nhwc2nchw)
=======
REGISTER_LITE_KERNEL(layout_once, kARM, kInt8, kNCHW, NHWC_int8, int8_nhwc2nchw)
>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kARM),
                                      PRECISION(kInt8),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kInt8),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
