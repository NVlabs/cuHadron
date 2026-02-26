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
 * kernel_memcpy_dtoh_race.cu — cudaMemcpyAsync(DtoH) vs kernel write race.
 *
 * A kernel writes to device memory while cudaMemcpyAsync copies that memory
 * to the host on another stream. The copy can read old or new values (read-write
 * race). FIXED adds synchronization so the kernel completes before the DtoH
 * copy is issued.
 *
 * Race pattern: DtoH memcpy reads device memory while kernel writes it.
 *
 * When FIXED is defined, kernel is synchronized before memcpy (no race).
 * When FIXED is not defined, kernel and memcpy run concurrently.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel that writes computed values to dst
__global__ void write_kernel(int* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Make kernel longer to increase chance of parallel execution
    // Use strided writes across the array to prevent optimization
    int sum = 0;
    int stride = (N / 32);  // Stride size
    for (int i = 0; i < 200; i++) {
        // Write to different locations in strided pattern
        int write_idx = (idx + (i * stride)) % N;
        dst[write_idx] = idx * 100 + i;
        sum += i;
    }
    
    // Final write - the value we'll check
    dst[idx] = idx * 100 + (sum % 10);
}

int main() {
#ifdef FIXED
    printf("Running asyncmemcpy/kernel_memcpy_dtoh_race (FIXED VERSION)\n");
#else
    printf("Running asyncmemcpy/kernel_memcpy_dtoh_race (RACE VERSION)\n");
#endif
    printf("Tests read-write races when cudaMemcpyAsync reads while kernel writes on different stream\n\n");
    
    // Use very large buffer to make memcpy take significant time
    const int N = 512 * 1024 * 1024;  // 512M elements = 2GB
    const int KERNEL_N = 64 * 1024 * 1024;  // Kernel only processes first 64M elements
    
    // Allocate page-locked host memory for async copy to work concurrently
    int* h_result;
    cudaMallocHost(&h_result, N * sizeof(int));
    
    // Initialize host with sentinel values
    for (int i = 0; i < N; i++) {
        h_result[i] = -999;  // Sentinel value
    }
    
    
    // Allocate device memory
    int* d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    
    // Initialize device memory with zero
    cudaMemset(d_data, 0, N * sizeof(int));
    
    // Create two non-blocking streams
    cudaStream_t stream1, stream2;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    
    // Force module loading before launching kernels to avoid serialization
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)write_kernel);

    // Launch kernel on stream1 to write to d_data
    // Kernel only processes first KERNEL_N elements (not entire buffer)
    int threads = 256;
    int blocks = (KERNEL_N + threads - 1) / threads;
    write_kernel<<<blocks, threads, 0, stream1>>>(d_data, KERNEL_N);
    
#ifdef FIXED
    // FIXED: Synchronize using cudaEvent to ensure kernel completes before memcpy
    cudaEvent_t sync_event;
    cudaEventCreate(&sync_event);
    cudaEventRecord(sync_event, stream1);  // Record when kernel completes
    cudaStreamWaitEvent(stream2, sync_event, 0);  // Memcpy waits for kernel
    cudaMemcpyAsync(h_result, d_data, N * sizeof(int), cudaMemcpyDeviceToHost, stream2);
    cudaEventDestroy(sync_event);
#else
    // RACE: Missing synchronization!
    // Should call cudaStreamSynchronize(stream1) or use cudaEvent to ensure kernel completes
    
    // Copy d_data from device to host on stream2 (reads entire 2GB buffer)
    // RACE: memcpy may read before kernel finishes writing
    cudaMemcpyAsync(h_result, d_data, N * sizeof(int), cudaMemcpyDeviceToHost, stream2);
#endif
    
    // Synchronize both streams
    cudaDeviceSynchronize();
    
    // Print results
    printf("Results after concurrent operations:\n\n");
    
    printf("=== write_kernel (stream1) ===\n");
    printf("Kernel wrote computed values (idx*100 + offset) to first %d elements\n", KERNEL_N);
    printf("Expected to complete before memcpy reads\n");
    printf("Status: Kernel completed ✓\n\n");
    
    printf("=== cudaMemcpyAsync DtoH (stream2) ===\n");
    printf("Memcpy reads entire 2GB buffer from device to host\n");
    printf("Checking all %d elements kernel wrote:\n", KERNEL_N);
    
    int saw_correct = 0;
    int saw_zero = 0;
    int saw_sentinel = 0;
    
    // Check all elements the kernel wrote
    for (int idx = 0; idx < KERNEL_N; idx++) {
        int expected = idx * 100;  // Approximate expected value (kernel writes idx*100 + small offset)
        int result = h_result[idx];
        
        // Check if result is close to expected (accounting for sum % 10 offset)
        if (result >= expected && result <= expected + 10) {
            saw_correct++;
        } else if (result == 0) {
            saw_zero++;
        } else if (result == -999) {
            saw_sentinel++;
        }
    }
    
    // Print summary and sample of problematic elements
    printf("\nSummary of all %d elements:\n", KERNEL_N);
    printf("  Correct values (kernel wrote before memcpy read): %d\n", saw_correct);
    printf("  Zero values (memcpy read before kernel wrote): %d\n", saw_zero);
    printf("  Sentinel values (memcpy never read): %d\n", saw_sentinel);
    
    if (saw_zero > 0 || saw_sentinel > 0) {
        printf("\nShowing first 32 elements with race:\n");
        printf("Index | Expected (kernel) | Memcpy Read | Status\n");
        printf("------|-------------------|-------------|--------\n");
        int shown = 0;
        for (int idx = 0; idx < KERNEL_N && shown < 32; idx++) {
            int expected = idx * 100;
            int result = h_result[idx];
            
            if (result == 0 || result == -999) {
                const char* status = (result == 0) ? "Read zero (race!)" : "Read sentinel (race!)";
                printf("%5d | %17d | %11d | %s\n", idx, expected, result, status);
                shown++;
            }
        }
    }
    printf("\n");
    
    // Check results
    printf("=== Results ===\n");
    if (saw_correct == KERNEL_N) {
        printf("Status: No race - kernel completed before memcpy read\n");
    } else if (saw_zero > 0 || saw_sentinel > 0) {
        printf("Status: Race detected - memcpy read d_data before kernel finished writing\n");
        printf("Race percentage: %.2f%% of elements read too early\n", 
               100.0 * (saw_zero + saw_sentinel) / KERNEL_N);
    }
    
    
    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data);
    cudaFreeHost(h_result);
    
    printf("Test complete\n");
    
    return 0;
}

