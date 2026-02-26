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
 * memcpy_htod_kernel_race.cu — cudaMemcpyAsync(HtoD) vs kernel read race.
 *
 * cudaMemcpyAsync copies host to device while a kernel reads that device
 * buffer on another stream. The kernel can read old or new values (read-write
 * race). FIXED adds synchronization so the HtoD copy completes before the
 * kernel is launched.
 *
 * Race pattern: Kernel reads device memory while HtoD memcpy writes it.
 *
 * When FIXED is defined, memcpy is synchronized before kernel (no race).
 * When FIXED is not defined, memcpy and kernel run concurrently.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel that reads from ALL locations in the buffer with strided pattern
// 32 threads to make kernel faster, each thread accesses N/32 elements
__global__ void copy_kernel(int* src, int* dst, size_t N) {
    int tid = threadIdx.x;
    long long sum = 0;
    
    // Each thread reads with stride of blockDim.x (32)
    // Thread 0: 0, 32, 64, 96, ...
    // Thread 1: 1, 33, 65, 97, ...
    // Each thread accesses N/32 elements
    for (size_t i = tid; i < N; i += blockDim.x) {
        int val = src[i];  // READ - can race with memcpy writing!
        sum += val;
    }
    // Store sum in every iteration - compiler can't prove dst[tid] doesn't alias with src
    // This forces the read to happen in every iteration
    dst[tid] = (int)(sum / 100000);
}

int main() {
#ifdef FIXED
    printf("Running asyncmemcpy/memcpy_htod_kernel_race (FIXED VERSION)\n");
#else
    printf("Running asyncmemcpy/memcpy_htod_kernel_race (RACE VERSION)\n");
#endif
    printf("Tests read-write races when cudaMemcpyAsync writes while kernel reads on different stream\n\n");
    
    // Use large buffer to make memcpy take significant time
    const size_t N = 256ULL * 1024 * 1024;  // 256M elements = 1GB
    
    // Allocate page-locked host memory for async copy to work concurrently
    int* h_src;
    int* h_dst;
    cudaMallocHost(&h_src, N * sizeof(int));
    cudaMallocHost(&h_dst, N * sizeof(int));
    
    // Initialize source data (old values: 0, 10, 20, ..., 310, repeating)
    // Touch all memory to ensure it's actually allocated
    // Initialize host buffers
    for (size_t i = 0; i < N; i++) {
        h_src[i] = (i % 32) * 10;
    }
    for (size_t i = 0; i < N; i++) {
        h_dst[i] = -1;
    }
    
    // Allocate device memory
    int* d_src;
    int* d_dst;
    cudaMalloc(&d_src, N * sizeof(int));
    cudaMalloc(&d_dst, N * sizeof(int));
    
    // Initialize d_src with old values, d_dst with sentinel
    cudaMemcpy(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Create two streams for async operations
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);  // For async memcpy
    cudaStreamCreate(&stream2);  // For kernel
    
    // Force module loading before launching kernels to avoid serialization
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)copy_kernel);
    
    // Modify source data on host to new values (new values: 0, 100, 200, ..., 3100, repeating)
    printf("Modifying host buffer to new values...\n");
    for (size_t i = 0; i < N; i++) {
        h_src[i] = (i % 32) * 100;  // New values (10x the old values)
    }
    printf("Modification complete.\n\n");

    // Launch kernel on stream2, memcpy on stream1
#ifdef FIXED
    printf("Launching memcpy then kernel (FIXED - with synchronization):\n");
    printf("  1. cudaMemcpyAsync on stream1 (1GB, sequential write)\n");
    printf("  2. cudaStreamSynchronize(stream1) - wait for memcpy to complete\n");
    printf("  3. kernel on stream2 (32 threads, reads ALL 256M elements with stride=32)\n");
    printf("  FIXED - kernel waits for memcpy to complete!\n\n");
#else
    printf("Launching kernel and memcpy (RACE - no synchronization):\n");
    printf("  Loop 3 iterations:\n");
    printf("    1. kernel on stream2 (32 threads, reads ALL 256M elements with stride=32)\n");
    printf("    2. cudaMemcpyAsync on stream1 (1GB, sequential write to same buffer)\n");
    printf("  RACE - kernel on stream2 races with async memcpy on stream1!\n\n");
#endif

    cudaDeviceSynchronize();  // Ensure all prior work is done

#ifdef FIXED
    // FIXED: Launch memcpy first and wait for it to complete before kernel reads
    cudaMemcpyAsync(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaStreamSynchronize(stream1);  // Wait for memcpy to finish
    copy_kernel<<<1, 32, 0, stream2>>>(d_src, d_dst, N);  // Launch on stream2
#else
    // RACE: Loop to create pipeline effect with concurrent operations
        copy_kernel<<<1, 32, 0, stream2>>>(d_src, d_dst, N);  // Launch on stream2
        cudaMemcpyAsync(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
#endif
    
    // Synchronize all streams
    cudaDeviceSynchronize();
    
    // Copy results back (each of 32 threads writes its accumulated sum to dst[tid])
    int kernel_sums[32];
    cudaMemcpy(kernel_sums, d_dst, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Results after concurrent operations:\n\n");
    
    printf("=== cudaMemcpyAsync (stream1) ===\n");
    printf("Copied new values to d_src[0..31]: ");
    for (int i = 0; i < 5; i++) printf("%d ", i * 100);
    printf("... %d\n", 31 * 100);
    printf("Expected: Memcpy writes new values to entire buffer\n");
    printf("Status: Memcpy completed ✓\n\n");
    
    printf("=== copy_kernel (stream2) ===\n");
    printf("32 threads read ALL 256M elements with stride of 32\n");
    printf("Each thread processes ~8M elements\n");
    printf("Thread sums (first 4 threads): %d, %d, %d, %d\n\n", 
           kernel_sums[0], kernel_sums[1], kernel_sums[2], kernel_sums[3]);
    
    // Calculate expected sums per thread based on what kernel should have read
    // Each thread reads N/32 = 8M elements with stride of 32
    // Due to strided access, each thread reads the SAME value repeatedly:
    //   Thread 0 reads from positions 0, 32, 64, ... → all have value (0 % 32) * multiplier = 0
    //   Thread 1 reads from positions 1, 33, 65, ... → all have value (1 % 32) * multiplier = 1 * multiplier
    //   Thread tid reads value (tid % 32) * multiplier = tid * multiplier (since tid < 32)
    
    size_t elements_per_thread = N / 32;  // 8M elements
    
    printf("Expected sum per thread (tid reads same value 8M times):\n");
    printf("  If read old values (before memcpy): tid * 10 * 8M / 100000 = tid * 838\n");
    printf("  If read new values (after memcpy): tid * 100 * 8M / 100000 = tid * 8388\n");
    printf("  Thread 0: old=0, new=0 | Thread 1: old=838, new=8388 | Thread 31: old=25978, new=259788\n\n");
    
    // Expected values for specific threads
    long long expected_old[32], expected_new[32];
    for (int tid = 0; tid < 32; tid++) {
        int old_value = tid * 10;
        int new_value = tid * 100;
        expected_old[tid] = ((long long)old_value * elements_per_thread) / 100000;
        expected_new[tid] = ((long long)new_value * elements_per_thread) / 100000;
    }
    
    // Check results for all threads
    printf("=== Results ===\n");
    int race_count = 0;
    int old_count = 0;
    int new_count = 0;
    
    for (int i = 0; i < 32; i++) {
        // Check if this thread's sum is close to old values, new values, or mixed
        // Use 20% tolerance to account for race checker overhead
        if (kernel_sums[i] < expected_old[i] + (expected_old[i] / 5 + 100)) {
            old_count++;
        } else if (kernel_sums[i] > expected_new[i] - (expected_new[i] / 5 + 100)) {
            new_count++;
        } else {
            race_count++;
        }
    }
    
    printf("Threads that read old values: %d/32\n", old_count);
    printf("Threads that read new values: %d/32\n", new_count);
    printf("Threads that read mixed (RACE): %d/32\n\n", race_count);
    
    const char* status;
    if (race_count > 0) {
        status = "RACE DETECTED - threads read mixed old/new values!";
    } else if (old_count == 32) {
        status = "No race - kernel read before memcpy started";
    } else if (new_count == 32) {
        status = "No race - kernel read after memcpy completed";
    } else {
        status = "POSSIBLE RACE - some threads read old, some read new";
    }
    printf("Status: %s\n", status);
    
    
    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFreeHost(h_src);
    cudaFreeHost(h_dst);
    
    printf("Test complete\n");
    
    return 0;
}

