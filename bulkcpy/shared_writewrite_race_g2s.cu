/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * shared_writewrite_race_g2s.cu — cp.async.bulk global-to-shared write-write race.
 *
 * Thread 0 uses cp.async.bulk to copy 256 bytes from global to shared. Other
 * threads read from shared. In race mode, they read before the bulk copy is
 * waited on, so they can see uninitialized or in-flight data (read-write race
 * with the bulk copy). FIXED adds cp.async.bulk.wait_group and __syncthreads.
 *
 * Race pattern: Threads 1–63 read shared_data before bulk copy completes.
 *
 * When FIXED is defined, wait_group and __syncthreads before other threads read.
 * When FIXED is not defined, other threads read without waiting for the copy.
 */

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void async_bulk_race_kernel(int* global_data, int* result) {
    __shared__ int shared_data[64];
    __shared__ unsigned long long mbar;
    unsigned int mbar_addr = __cvta_generic_to_shared(&mbar);
   
    int tid = threadIdx.x;
    
    // Initialize shared memory to sentinel values
    shared_data[tid] = -1;
    
    // Initialize mbarrier
    if (tid == 0) {
        unsigned int mbar_addr = __cvta_generic_to_shared(&mbar);
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mbar_addr));
        // Expect 256 bytes for the bulk copy
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], 256;" :: "r"(mbar_addr));
    }
    __syncthreads();
    
    // Thread 0 uses cp.async.bulk to copy from global to shared memory
    if (tid == 0) {
        // Bulk copy 256 bytes (64 ints) from global[0] to shared[0]
        unsigned int shared_addr = __cvta_generic_to_shared(&shared_data[0]);
        unsigned int mbar_addr = __cvta_generic_to_shared(&mbar);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 256, [%2];" 
                     :: "r"(shared_addr),
                        "l"(&global_data[0]),
                        "r"(mbar_addr));
    }
#ifdef FIXED
    // FIXED: Wait for bulk copy to complete before other threads write to shared
    unsigned long long state = 0;
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "try_wait_loop:\n\t"
                 "mbarrier.try_wait.shared.b64 p, [%0], %1;\n\t"
                 "@!p bra try_wait_loop;\n\t"
                 "}" :: "r"(mbar_addr), "l"(state));
#endif
    // All threads immediately try to write to shared memory
    // This creates a race! Thread 0's bulk copy is writing while others are writing
    shared_data[tid] = 900 + tid;
    
    // Store what's in shared (racing value) for verification
    result[tid] = shared_data[tid];
}

int main() {
#ifdef FIXED
    printf("Running bulkcpy/shared_writewrite_race_g2s (FIXED VERSION)\n");
#else
    printf("Running bulkcpy/shared_writewrite_race_g2s (RACE VERSION)\n");
#endif
    printf("Tests write-write races in shared memory when bulk copy writes while other threads write (global to shared)\n\n");
    
    const int N = 64;
    int h_data[N];
    int h_result[N];
    
    // Initialize global data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 10;
    }
    
    // Allocate device memory
    int* d_data;
    int* d_result;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_result, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    async_bulk_race_kernel<<<1, 64>>>(d_data, d_result);
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_result, d_result, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_result);
    
    printf("Test complete\n");
    
    return 0;
}

