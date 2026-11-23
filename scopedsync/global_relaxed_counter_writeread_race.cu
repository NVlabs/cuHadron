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


// Handoff through an atomic counter: the producer writes the values and increments a counter with a relaxed atomicAdd;
// the consumer spins until the counter is set (relaxed atomic reads) and reads the values. Relaxed RMWs do not order
// the values. FIXED: atom.add.release.gpu and an acquire spin (ld.acquire.gpu).
#include <cuda_runtime.h>
#include <stdio.h>
#define TEST_NAME "scopedsync/global_relaxed_counter_writeread_race"
#define TEST_DESC "Tests a data race through a relaxed atomic counter handoff between thread blocks"

__global__ void kernel(unsigned* buf) {
    unsigned* counter = buf; unsigned* data = buf + 32; unsigned* out = buf + 1024;
    if (blockIdx.x == 1) {
        data[threadIdx.x] = threadIdx.x + 1;
        __syncwarp();
        if (threadIdx.x == 0) {
#ifdef FIXED
            asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(counter) : "memory");
#else
            atomicAdd(counter, 1u);
#endif
        }
    } else {
        if (threadIdx.x == 0) {
            unsigned v = 0;
            do {
#ifdef FIXED
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(counter) : "memory");
#else
                v = atomicAdd(counter, 0u);
#endif
            } while (v == 0);
        }
        __syncwarp();
        out[threadIdx.x] = data[threadIdx.x];
    }
}
#include "common_main.inc"
