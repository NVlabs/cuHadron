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


// CUDA's threadFenceReduction pattern without its fence: every block writes a partial result, then takes a ticket
// with atomicAdd; the block that takes the last ticket sums the partials. Without __threadfence() before the ticket,
// the last block's reads of the partials race with the other blocks' writes. FIXED: the fence before the ticket.
#include <cuda_runtime.h>
#include <stdio.h>
#define TEST_NAME "scopedsync/global_last_block_reduction_writeread_race"
#define TEST_DESC "Tests a data race in a last-block reduction whose writers skip __threadfence()"
#define BLOCKS 64

__global__ void reduce_kernel(unsigned* partial, unsigned* ticket, unsigned* out) {
    __shared__ bool last;
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = blockIdx.x + 1;
#ifdef FIXED
        __threadfence();
#endif
        last = atomicAdd(ticket, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (last && threadIdx.x == 0) {
        __threadfence();
        unsigned s = 0;
        for (int i = 0; i < gridDim.x; i++) s += ((volatile unsigned*)partial)[i];
        *out = s;
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
    cudaMalloc(&d, 1024 * sizeof(unsigned));
    cudaMemset(d, 0, 1024 * sizeof(unsigned));
    reduce_kernel<<<BLOCKS, 32>>>(d, d + 512, d + 513);
    cudaDeviceSynchronize();
    unsigned s = 0;
    cudaMemcpy(&s, d + 513, sizeof s, cudaMemcpyDeviceToHost);
    printf("sum %u (expected %u)\n", s, BLOCKS * (BLOCKS + 1) / 2);
    cudaFree(d);
    printf("Test complete\n");
    return 0;
}
