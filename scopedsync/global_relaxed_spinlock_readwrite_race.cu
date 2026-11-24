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


// A spinlock between blocks built from atomicCAS/atomicExch with no fences: both blocks increment a plain counter
// inside the critical section. Lock acquire and release are relaxed atomics, so the critical sections are not ordered
// and the counter accesses race. FIXED: __threadfence() after taking the lock and before releasing it.
#include <cuda_runtime.h>
#include <stdio.h>
#define TEST_NAME "scopedsync/global_relaxed_spinlock_readwrite_race"
#define TEST_DESC "Tests a data race inside a spinlock whose acquire and release have no fences"

__global__ void lock_kernel(unsigned* lock, unsigned* counter) {
    if (threadIdx.x != 0) return;
    for (int i = 0; i < 100; i++) {
        while (atomicCAS(lock, 0u, 1u) != 0u) {}
#ifdef FIXED
        __threadfence();
#endif
        *counter = *counter + 1;
#ifdef FIXED
        __threadfence();
#endif
        atomicExch(lock, 0u);
    }
}

int main() {
#ifdef FIXED
    printf("Running %s (FIXED VERSION)\n", TEST_NAME);
#else
    printf("Running %s (RACE VERSION)\n", TEST_NAME);
#endif
    printf("%s\n\n", TEST_DESC);
    unsigned *d;
    cudaMalloc(&d, 64 * sizeof(unsigned));
    cudaMemset(d, 0, 64 * sizeof(unsigned));
    lock_kernel<<<2, 32>>>(d, d + 32);
    cudaDeviceSynchronize();
    unsigned c = 0;
    cudaMemcpy(&c, d + 32, sizeof c, cudaMemcpyDeviceToHost);
    printf("counter %u (expected 200)\n", c);
    cudaFree(d);
    printf("Test complete\n");
    return 0;
}
