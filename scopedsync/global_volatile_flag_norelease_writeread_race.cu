// SPDX-FileCopyrightText: Copyright (c) 2026 Shreshth Rajan. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// The other half: the consumer fences after its volatile spin, but the producer sets the volatile flag right after
// writing the values, with no fence in between. The flag store is not a release, so the reads race with the writes.
// FIXED: __threadfence() before the flag store.
#include <cuda_runtime.h>
#include <stdio.h>
#define TEST_NAME "scopedsync/global_volatile_flag_norelease_writeread_race"
#define TEST_DESC "Tests a data race through a volatile flag with a consumer fence but no producer fence"

__global__ void kernel(unsigned* buf) {
    volatile unsigned* flag = buf; unsigned* data = buf + 32; unsigned* out = buf + 1024;
    if (blockIdx.x == 1) {
        data[threadIdx.x] = threadIdx.x + 1;
#ifdef FIXED
        __threadfence();
#endif
        __syncwarp();
        if (threadIdx.x == 0) *flag = 1;
    } else {
        if (threadIdx.x == 0) while (*flag == 0) {}
        __threadfence();
        __syncwarp();
        out[threadIdx.x] = data[threadIdx.x];
    }
}
#include "common_main.inc"
