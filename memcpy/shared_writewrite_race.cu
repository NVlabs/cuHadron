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
 * shared_writewrite_race.cu — cp.async shared-memory write-write race.
 *
 * Threads use cp.async to copy from global to shared memory. In race mode,
 * shared memory has 32 elements and threads use tid % 32, so multiple warps
 * write to the same locations (write-write race). In FIXED mode, 64 elements
 * and no modulo give each thread a unique index (no race).
 *
 * Race pattern: Threads in different warps write to shared_data[threadIdx.x % 32].
 *
 * When FIXED is defined, each thread has a unique shared index (no race).
 * When FIXED is not defined, 32 elements and modulo create write-write conflicts.
 */

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void data_race_kernel(int* global_data) {
#ifdef FIXED
    // FIXED: Use 64 elements to avoid write-write conflicts
    __shared__ int shared_data[64];
    int shared_idx = threadIdx.x;  // No modulo - each thread gets unique index
#else
    // RACE: Use 32 elements with modulo - creates write-write conflicts
    __shared__ int shared_data[32];
    // Thread 0 -> index 0, thread 1 -> index 1, ..., thread 31 -> index 31
    // Thread 32 -> index 0, thread 33 -> index 1, ..., thread 63 -> index 31
    // This creates 32 race conditions between corresponding threads in each warp!
    int shared_idx = threadIdx.x % 32;
#endif
    
    int tid = threadIdx.x;
    
    // Use cp.async via inline PTX to copy from global to shared memory
    // ALL threads participate - each copies to its computed address
    unsigned int shared_addr = __cvta_generic_to_shared(&shared_data[shared_idx]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" 
                 :: "r"(shared_addr), 
                    "l"(&global_data[tid]));
    
    // Wait for all async copies to complete
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();
    
    // Copy back from shared to global using computed address
    // ALL threads participate
    global_data[tid] = shared_data[shared_idx];
}

int main() {
#ifdef FIXED
    printf("Running memcpy/shared_writewrite_race (FIXED VERSION)\n");
#else
    printf("Running memcpy/shared_writewrite_race (RACE VERSION)\n");
#endif
    printf("Tests write-write races in shared memory using cp.async with multiple threads copying to same address\n\n");
    
    const int N = 64;
    int h_data[N];
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 10;
    }
    
    // Allocate device memory
    int* d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel with 1 block, 64 threads (2 warps)
    data_race_kernel<<<1, 64>>>(d_data);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_data);
    
    printf("Test complete\n");
    
    return 0;
}

