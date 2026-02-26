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
 * global_readwrite_race.cu — CPU–GPU read-write race on unified memory.
 *
 * A GPU kernel writes to unified memory; the host reads the result. In race
 * mode, the host reads before the kernel completes (missing
 * cudaDeviceSynchronize), so the read can see partial or old data (read-write
 * race). FIXED adds cudaDeviceSynchronize before the host reads.
 *
 * Race pattern: CPU reads unified memory before GPU kernel completes.
 *
 * When FIXED is defined, host synchronizes before reading (no race).
 * When FIXED is not defined, host reads without synchronizing.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Producer kernel: Computes and writes results
__global__ void compute_kernel(int* data) {
    int tid = threadIdx.x;
    // Add computation delay to increase race window
    int temp = 0;
    for (int i = 0; i < 10000000; i++) {
        temp += i;
    }
    // Write simple value after delay
    data[tid] = tid * 100 + temp % 10;
}

int main() {
#ifdef FIXED
    printf("Running hostdevice/global_readwrite_race (FIXED VERSION)\n");
#else
    printf("Running hostdevice/global_readwrite_race (RACE VERSION)\n");
#endif
    printf("Tests read-write races when CPU reads unified memory before GPU kernel completes\n\n");
    
    const int N = 32;
    int* data;
    int cpu_results[N];
    
    // Allocate unified memory accessible by both CPU and GPU
    cudaMallocManaged(&data, N * sizeof(int));
    
    // Initialize data to sentinel values
    for (int i = 0; i < N; i++) {
        data[i] = -999;
    }
    
    // Initialize data
    
    // Launch kernel to compute results
    compute_kernel<<<1, 32>>>(data);
    
    // MISSING SYNCHRONIZATION!
    // Should call cudaDeviceSynchronize() here
    
    // RACE: CPU reads results before GPU kernel completes
    for (int i = 0; i < N; i++) {
        cpu_results[i] = data[i];  // Process GPU results
    }
    
    // Synchronize (too late - race already occurred)
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(data);

    printf("Test complete (read %d values, first=%d)\n", N, cpu_results[0]);
    
    return 0;
}

