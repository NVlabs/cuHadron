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
 * shared_bytewise_readwrite.cu — False-positive check: byte-level independence (shared).
 *
 * Thread 0 reads byte 0 of a 4-byte word in shared memory; thread 32 writes
 * byte 1. There is no data race; the accesses are to different bytes. A
 * detector that reports a race here produces a false positive.
 *
 * Pattern: Thread 0 reads byte 0, thread 32 writes byte 1 of the same int.
 *
 * No FIXED variant; the test always has no race (byte-level independence).
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel where one thread reads a byte while another writes a different byte
__global__ void bytewise_readwrite_kernel(int* results) {
    __shared__ int shared_data;
    
    int tid = threadIdx.x;
    
    // Initialize to known value
    if (tid == 0) {
        shared_data = 0x11223344;  // Known initial value
    }
    __syncthreads();
    
    // RACE: Thread 0 reads byte 0, Thread 32 writes byte 1
    // Reading and writing different bytes of the SAME 4-byte word
    if (tid == 0) {
        // Warp 0, thread 0: Read from byte 0
        char* byte_ptr = (char*)&shared_data;
        unsigned char read_value = byte_ptr[0];  // Read 0x44
        results[0] = read_value;
    }
    else if (tid == 32) {
        // Warp 1, thread 0: Write to byte 1
        char* byte_ptr = (char*)&shared_data;
        byte_ptr[1] = 0xAA;  // Write 0xAA to second byte
    }
    
    // Wait for both operations to complete
    __syncthreads();
    
    // Read final value
    if (tid == 0) {
        results[1] = shared_data;
    }
}

int main() {
    printf("Running false_positives/shared_bytewise_readwrite (false positive check, no race)\n");
    printf("Tests byte-level granularity - different bytes in same word (false positive check)\n\n");
    
    int* results;
    
    cudaMallocManaged(&results, 2 * sizeof(int));
    results[0] = 0;
    results[1] = 0;
    
    // Launch kernel with 1 block, 64 threads (2 warps)
    bytewise_readwrite_kernel<<<1, 64>>>(results);
    cudaDeviceSynchronize();
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("Execution completed.\n\n");
    
    // Print results
    unsigned char read_value = results[0] & 0xFF;
    int final_value = results[1];
    
    printf("Thread 0 read value: 0x%02X (expected 0x44)\n", read_value);
    printf("Final value: 0x%08X\n", final_value);
    
    unsigned char byte0 = (final_value >> 0) & 0xFF;
    unsigned char byte1 = (final_value >> 8) & 0xFF;
    unsigned char byte2 = (final_value >> 16) & 0xFF;
    unsigned char byte3 = (final_value >> 24) & 0xFF;
    
    printf("\nFinal bytes:\n");
    printf("Byte 0: 0x%02X (expected 0x44, unchanged)\n", byte0);
    printf("Byte 1: 0x%02X (expected 0xAA, written by thread 32)\n", byte1);
    printf("Byte 2: 0x%02X (expected 0x22, unchanged)\n", byte2);
    printf("Byte 3: 0x%02X (expected 0x11, unchanged)\n", byte3);
    printf("\n");
    
    printf("=== Analysis ===\n");
    printf("This test has one thread READ a byte while another WRITES a different byte.\n");
    printf("Race detectors must determine if this is a race:\n");
    printf("  - Word-level granularity: YES, this is a race (both access same word)\n");
    printf("  - Byte-level granularity: NO, this is not a race (different bytes)\n");
    printf("\n");
    printf("Root Cause: Byte-level read-write to same word\n");
    printf("            Thread 0 reads byte 0, Thread 32 writes byte 1\n");
    printf("            Missing __syncthreads() between initialization and operations\n");
    
    // Cleanup
    cudaFree(results);
    
    printf("Test complete\n");
    
    return 0;
}

