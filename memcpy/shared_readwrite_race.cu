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
 * shared_readwrite_race.cu — cp.async shared-memory read-write race.
 *
 * Threads use cp.async to copy from global to shared memory, then read from
 * shared. In race mode, the read happens before cp.async.wait_all, so the
 * read can see uninitialized or in-flight data (read-write race with the
 * async copy). In FIXED mode, wait_all is called before reading.
 *
 * Race pattern: Read from shared_data[tid] before cp.async completes.
 *
 * When FIXED is defined, cp.async.wait_all is used before reading (no race).
 * When FIXED is not defined, the read may happen before the copy completes.
 */

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void async_race_kernel(int* global_data, int* result) {
    __shared__ int shared_data[64];
    
    int tid = threadIdx.x;
    
    // Each thread uses cp.async to copy from global to shared memory
    unsigned int shared_addr = __cvta_generic_to_shared(&shared_data[tid]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" 
                 :: "r"(shared_addr), 
                    "l"(&global_data[tid]));
    
#ifdef FIXED
    // FIXED: Wait for cp.async to complete before reading
    asm volatile("cp.async.wait_all;\n");
    
    int value = shared_data[tid];
    
    // Store the correct value
    result[tid] = value;
    result[tid + 64] = shared_data[tid];
#else
    // RACE: Missing synchronization! Should call cp.async.wait_all here
    // Race: cp.async might not have completed yet when we read
    
    // Immediately try to read from shared memory
    // This creates a race! We might read uninitialized data or the copied data
    int value = shared_data[tid];
    
    // Now wait (too late!) 
    asm volatile("cp.async.wait_all;\n");
    
    // Store what we read (racing value) and what the correct value should be
    result[tid] = value;                    // What we actually read (race)
    result[tid + 64] = shared_data[tid];    // What it should be (after wait)
#endif
}

int main() {
#ifdef FIXED
    printf("Running memcpy/shared_readwrite_race (FIXED VERSION)\n");
#else
    printf("Running memcpy/shared_readwrite_race (RACE VERSION)\n");
#endif
    printf("Tests read-write races in shared memory when reading before cp.async completes\n\n");
    
    const int N = 64;
    int h_data[N];
    int h_result[N * 2];
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 10;
    }
    
    // Allocate device memory
    int* d_data;
    int* d_result;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_result, N * 2 * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    async_race_kernel<<<1, 64>>>(d_data, d_result);
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_result, d_result, N * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_result);
    
    printf("Test complete\n");
    
    return 0;
}

