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
 * global_writewrite_race.cu — cp.async.bulk global write-write race (two blocks).
 *
 * Two blocks each do bulk copies from their shared memory to global. In race
 * mode, both blocks write to global[0..31], causing a write-write race. In
 * FIXED mode, block 0 writes global[0..31] and block 1 writes global[32..63].
 *
 * Race pattern: Block 0 and block 1 both bulk-copy to global[0..31].
 *
 * When FIXED is defined, each block writes a distinct global region (no race).
 * When FIXED is not defined, both blocks write the same global region.
 */

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void global_bulk_race_kernel(int* global_data) {
    __shared__ int shared_data[32];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Initialize shared memory differently per block
    // Block 0: shared[i] = i * 10
    // Block 1: shared[i] = 900 + i
    if (bid == 0) {
        shared_data[tid] = tid * 10;
    } else {
        shared_data[tid] = 900 + tid;
    }
    __syncthreads();
      
    // Thread 0 in EACH block does bulk copies in a loop to increase race window
    if (tid == 0) {
#ifdef FIXED
        // FIXED: Each block writes to different global address
        // Block 0 writes to global[0:31], Block 1 writes to global[32:63]
        int* dest_addr = &global_data[bid * 32];
#else
        // RACE: Both blocks write to the SAME global address global[0:31]
        // THIS CREATES THE RACE! Both blocks' bulk copies target the same memory
        int* dest_addr = &global_data[0];
#endif
        
        // Loop to increase race window and conflicts
        for (int i = 0; i < 100; i++) {
            // Update shared memory to create varying values
            if (bid == 0) {
                shared_data[0] = tid * 10 + i;
            } else {
                shared_data[0] = 900 + tid + i;
            }
            
            // Bulk copy 128 bytes (32 ints) from shared[0] to global
            unsigned int shared_addr = __cvta_generic_to_shared(&shared_data[0]);
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 128;" 
                         :: "l"(dest_addr), 
                            "r"(shared_addr));
            asm volatile("cp.async.bulk.commit_group;");
            // Wait for THIS block's bulk copy to complete (doesn't sync with other blocks!)
            asm volatile("cp.async.bulk.wait_group 0;");
        }
    }
}

int main() {
#ifdef FIXED
    printf("Running bulkcpy/global_writewrite_race (FIXED VERSION)\n");
#else
    printf("Running bulkcpy/global_writewrite_race (RACE VERSION)\n");
#endif
    printf("Tests write-write races when multiple blocks use bulk copy to write to same global memory location\n\n");
    
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
    
    // Launch kernel with 2 blocks, 32 threads each
    global_bulk_race_kernel<<<2, 32>>>(d_data);
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_data);
    
    printf("Test complete\n");
    
    return 0;
}

