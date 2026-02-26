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
 * shared_readwrite_race.cu — cp.async.bulk shared read-write race (copy vs writers).
 *
 * Thread 0 bulk-copies shared memory to global. Threads 32–63 write to the
 * same shared region. In race mode, the copy reads shared while those threads
 * write (read-write race). FIXED adds commit_group, wait_group.read, and
 * __syncthreads before the writers run.
 *
 * Race pattern: Thread 0 reads shared[0..63] via bulk copy while threads 32–63 write it.
 *
 * When FIXED is defined, synchronization ensures copy completes before writes.
 * When FIXED is not defined, writers run without waiting for the copy.
 */

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void shared_bulk_race_kernel(int* global_data) {
    __shared__ int shared_data[64];
    
    int tid = threadIdx.x;
    
    // Initialize shared memory with initial values
    shared_data[tid] = tid * 10;
    __syncthreads();
    
    // RACE: Thread 0 reads from shared memory while threads 32-63 overwrite it
    // Thread 0: Bulk copy from shared[0:63] to global[0:63] - READING shared memory
    // Threads 32-63: WRITE to shared[tid - 32] - OVERWRITING the shared memory being read!
    
    if (tid == 0) {
        // Thread 0: Bulk copy 256 bytes (64 ints) from shared[0] to global[0]
        // This READ of shared memory races with writes from threads 32-63
        unsigned int shared_addr = __cvta_generic_to_shared(&shared_data[0]);
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 256;" 
                     :: "l"(&global_data[0]), 
                        "r"(shared_addr));
    }
    
#ifndef FIXED
    // RACE: Missing barrier here!
#else
    // FIXED: Add synchronization to ensure bulk copy completes before writes
    asm volatile("cp.async.bulk.commit_group;");
    asm volatile("cp.async.bulk.wait_group.read 0;");
    __syncthreads();
#endif
    
    // Threads 32-63: WRITE to shared memory that thread 0 is copying from
    if (tid >= 32 && tid < 64) {
        int shared_idx = tid - 32;
        shared_data[shared_idx] = 900 + tid;
    }
}

int main() {
#ifdef FIXED
    printf("Running bulkcpy/shared_readwrite_race (FIXED VERSION)\n");
#else
    printf("Running bulkcpy/shared_readwrite_race (RACE VERSION)\n");
#endif
    printf("Tests read-write races when bulk copy reads from shared memory while other threads write\n\n");
    
    const int N = 64;
    int h_data[N];
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = -1;
    }
    
    // Allocate device memory
    int* d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    shared_bulk_race_kernel<<<1, 64>>>(d_data);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_data);
    
    printf("Test complete\n");
    
    return 0;
}

