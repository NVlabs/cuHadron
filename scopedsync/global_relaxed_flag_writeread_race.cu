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


// Message passing between two blocks with a relaxed flag: the producer writes 32 values, then sets the flag with a
// relaxed store; the consumer spins on the flag with relaxed loads, then reads the values. No release/acquire orders
// the values before the flag (PTX memory model), so the consumer's reads race with the producer's writes.
// FIXED: st.release.gpu on the flag and ld.acquire.gpu in the spin.
#include <cuda_runtime.h>
#include <stdio.h>
#define TEST_NAME "scopedsync/global_relaxed_flag_writeread_race"
#define TEST_DESC "Tests a data race through a relaxed flag between thread blocks (no release/acquire)"

__device__ void set_flag(unsigned* f) {
#ifdef FIXED
    asm volatile("st.release.gpu.global.u32 [%0], 1;" :: "l"(f) : "memory");
#else
    asm volatile("st.relaxed.gpu.global.u32 [%0], 1;" :: "l"(f) : "memory");
#endif
}
__device__ unsigned get_flag(unsigned* f) {
    unsigned v;
#ifdef FIXED
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
#else
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
#endif
    return v;
}

__global__ void kernel(unsigned* buf) {
    unsigned* flag = buf; unsigned* data = buf + 32; unsigned* out = buf + 1024;
    if (blockIdx.x == 1) {
        data[threadIdx.x] = threadIdx.x + 1;
        __syncwarp();
        if (threadIdx.x == 0) set_flag(flag);
    } else {
        if (threadIdx.x == 0) while (get_flag(flag) == 0) {}
        __syncwarp();
        out[threadIdx.x] = data[threadIdx.x];
    }
}
#include "common_main.inc"
