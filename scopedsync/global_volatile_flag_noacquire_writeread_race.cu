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


// The classic "volatile flag" handoff with only half the fences: the producer writes the values, __threadfence(), then
// sets a volatile flag; the consumer spins on the volatile flag and reads the values without a fence after the spin.
// Without the consumer's fence the spin load is not an acquire, so the reads race with the writes.
// FIXED: __threadfence() after the spin.
#include <cuda_runtime.h>
#include <stdio.h>
#define TEST_NAME "scopedsync/global_volatile_flag_noacquire_writeread_race"
#define TEST_DESC "Tests a data race through a volatile flag with a producer fence but no consumer fence"

__global__ void kernel(unsigned* buf) {
    volatile unsigned* flag = buf; unsigned* data = buf + 32; unsigned* out = buf + 1024;
    if (blockIdx.x == 1) {
        data[threadIdx.x] = threadIdx.x + 1;
        __threadfence();
        __syncwarp();
        if (threadIdx.x == 0) *flag = 1;
    } else {
        if (threadIdx.x == 0) while (*flag == 0) {}
#ifdef FIXED
        __threadfence();
#endif
        __syncwarp();
        out[threadIdx.x] = data[threadIdx.x];
    }
}
#include "common_main.inc"
