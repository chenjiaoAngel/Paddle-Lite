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

#pragma once
<<<<<<< HEAD
=======

>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
<<<<<<< HEAD

class NCHWToNHWCCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
=======
template <PrecisionType Ptype>
class NCHWToNHWCCompute : public KernelLite<TARGET(kARM), Ptype> {
>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf
 public:
  using param_t = operators::LayoutParam;
  void Run() override;
  virtual ~NCHWToNHWCCompute() = default;
};

<<<<<<< HEAD
class NCHWToNHWCComputeInt8
    : public KernelLite<TARGET(kARM), PRECISION(kInt8)> {
 public:
  using param_t = operators::LayoutParam;
  void Run() override;
  virtual ~NCHWToNHWCComputeInt8() = default;
};

class NHWCToNCHWCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
=======
template <PrecisionType Ptype>
class NHWCToNCHWCompute : public KernelLite<TARGET(kARM), Ptype> {
>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf
 public:
  using param_t = operators::LayoutParam;
  void Run() override;
  virtual ~NHWCToNCHWCompute() = default;
};

<<<<<<< HEAD
class NHWCToNCHWComputeInt8
    : public KernelLite<TARGET(kARM), PRECISION(kInt8)> {
 public:
  using param_t = operators::LayoutParam;
  void Run() override;
  virtual ~NHWCToNCHWComputeInt8() = default;
};

=======
>>>>>>> d5b08275c46b2517790d170a469006246f59b6bf
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
