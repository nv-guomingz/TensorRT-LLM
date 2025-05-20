/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <pybind11/pybind11.h>

< < < < < < < < HEAD : cpp / tensorrt_llm / pybind / runtime / moeBindings.h namespace tensorrt_llm::pybind::runtime
{

    void initMoeBindings(pybind11::module_ & m);

} // namespace tensorrt_llm::pybind::runtime
== == == ==
#include "tensorrt_llm/common/cudaUtils.h"

    namespace tensorrt_llm::kernels::dsv3MinLatencyKernels
{

    template <typename T, int kNumTokens, int kHdIn, int kHdOut>
    void invokefusedAGemm(T * output, T const* mat_a, T const* mat_b, cudaStream_t const stream);

} // namespace tensorrt_llm::kernels::dsv3MinLatencyKernels
>>>>>>>> 6c8baded24(refactor dsr1_minlatency kernels)
    : cpp / tensorrt_llm / kernels / dsv3MinLatencyKernels / dsv3FusedAGemm.h
