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
 * global_readwrite_race.cu — Multi-GPU read-write race on unified memory.
 *
 * One GPU runs a writer kernel; another runs a reader kernel, both accessing
 * the same unified memory. Without synchronization between GPUs, the reader
 * can see partial or old data (read-write race). FIXED adds
 * cudaDeviceSynchronize on both devices and ensures the writer completes
 * before the reader runs. Requires at least two GPUs.
 *
 * Race pattern: GPU 1 reads unified memory while GPU 0’s kernel writes it.
 *
 * When FIXED is defined, host synchronizes so writer completes before reader.
 * When FIXED is not defined, reader may run while writer is still writing.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Writer kernel: Computes and writes results
__global__ void writer_kernel(int* data, int gpu_id, int* dummy) {
    int tid = threadIdx.x;
    int value = tid * tid;
    
    // Keep computing and writing frequently to increase race window
    for (int i = 0; i < 50000000; i++) {
        value = (value + i) % 1000000;
        data[tid] = value;
        
        // Each thread writes to its own dummy location to avoid intra-kernel race
        // Compiler can't prove dummy[tid] doesn't alias with data
        // This forces the write to happen in every iteration
        dummy[tid] = i;
    }
    // Final write
    data[tid] = tid * tid + (value % 10);
}

// Reader kernel: Reads from data
__global__ void reader_kernel(int* data, int* results) {
    int tid = threadIdx.x;
    long long sum = 0;
    
    // Keep reading frequently to increase race window
    // Use strided reads to make each read unique
    for (int i = 0; i < 1000000; i++) {
        // Strided read - access different location each iteration
        int idx = (tid + i) % 32;
        int val = data[idx];
        sum += val;
        
        // Store sum in every iteration - compiler can't prove results[tid] doesn't alias with data
        // This forces the read to happen in every iteration
        results[tid] = (int)(sum / 1000);
    }
}

int main() {
#ifdef FIXED
    printf("Running multigpu/global_readwrite_race (FIXED VERSION)\n");
#else
    printf("Running multigpu/global_readwrite_race (RACE VERSION)\n");
#endif
    printf("Tests read-write races when one GPU reads unified memory while another GPU writes\n\n");
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount < 2) {
        printf("This test requires at least 2 GPUs, found %d\n", deviceCount);
        return 1;
    }
    
    // Enable peer access between GPUs
    cudaSetDevice(0);
    cudaError_t err = cudaDeviceEnablePeerAccess(1, 0);
    
    cudaSetDevice(1);
    err = cudaDeviceEnablePeerAccess(0, 0);
    if (err == cudaSuccess) {
        printf("Enabled peer access: GPU 1 -> GPU 0\n");
    } else if (err != cudaErrorPeerAccessAlreadyEnabled) {
        printf("Warning: Could not enable peer access GPU 1 -> GPU 0: %s\n", cudaGetErrorString(err));
    }
    printf("\n");
    
    const int DATA_SIZE = 256;
    const int RESULT_SIZE = 256;
    
    // Allocate unified memory accessible by all GPUs
    int* data;
    int* results;
    int* dummy;
    cudaMallocManaged(&data, DATA_SIZE * sizeof(int));
    cudaMallocManaged(&results, RESULT_SIZE * sizeof(int));
    cudaMallocManaged(&dummy, 32 * sizeof(int));  // 32 threads need 32 dummy locations
    
    // Initialize data to sentinel values
    for (int i = 0; i < DATA_SIZE; i++) {
        data[i] = -999;
    }
    for (int i = 0; i < RESULT_SIZE; i++) {
        results[i] = 0;
    }
    for (int i = 0; i < 32; i++) {
        dummy[i] = 0;
    }
    
    // Create non-blocking streams for each GPU to enable concurrent execution
    cudaStream_t stream0, stream1;
    
    cudaSetDevice(0);
    cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking);
    
    cudaSetDevice(1);
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    
    // Force module loading before launching kernels to avoid serialization
    cudaFuncAttributes attr;
    cudaSetDevice(0);
    cudaFuncGetAttributes(&attr, (const void*)writer_kernel);
    cudaSetDevice(1);
    cudaFuncGetAttributes(&attr, (const void*)reader_kernel);
    
    printf("Launching kernels on different GPUs (32 threads each)...\n");
    
    // Launch writer kernel on GPU 0 - 32 threads
    cudaSetDevice(0);
    writer_kernel<<<1, 32, 0, stream0>>>(data, 0, dummy);
    
#ifdef FIXED
    // FIXED: Properly synchronize using event to ensure GPU 1 waits for GPU 0
    cudaEvent_t sync_event;
    cudaSetDevice(0);
    cudaEventCreate(&sync_event);
    cudaEventRecord(sync_event, stream0);  // Record when GPU 0 write completes
    
    cudaSetDevice(1);
    cudaStreamWaitEvent(stream1, sync_event, 0);  // GPU 1 waits for GPU 0
    reader_kernel<<<1, 32, 0, stream1>>>(data, results);
    
    cudaEventDestroy(sync_event);
#else
    // RACE: Launch reader kernel on GPU 1 - MISSING SYNCHRONIZATION!
    // Should wait for GPU 0 to finish writing
    cudaSetDevice(1);
    reader_kernel<<<1, 32, 0, stream1>>>(data, results);
#endif
    
    // Wait for both GPUs to complete
    cudaSetDevice(0);
    cudaStreamSynchronize(stream0);
    
    cudaSetDevice(1);
    cudaStreamSynchronize(stream1);
    
    printf("Execution completed.\n\n");
    
    // Check for errors
    cudaError_t err0, err1;
    cudaSetDevice(0);
    err0 = cudaGetLastError();
    
    cudaSetDevice(1);
    err1 = cudaGetLastError();
    
    if (err0 != cudaSuccess) {
        printf("GPU 0 Error: %s\n", cudaGetErrorString(err0));
    }
    if (err1 != cudaSuccess) {
        printf("GPU 1 Error: %s\n", cudaGetErrorString(err1));
    }
    
    // Analyze results
    printf("\n=== Final Results ===\n");
    printf("Data written by GPU 0 (first 10):\n");
    for (int i = 0; i < 10; i++) {
        printf("data[%d] = %d\n", i, data[i]);
    }
    
    printf("\nReader results from GPU 1 (first 10 threads):\n");
    for (int i = 0; i < 10; i++) {
        printf("Thread %d accumulated sum: %d\n", i, results[i]);
    }
    
    printf("\n=== Race Detection Summary ===\n");
    printf("GPU 0 (writer): 32 threads wrote to data[0..31] (50M iterations)\n");
    printf("GPU 1 (reader): 32 threads read from data[0..31] with strided access (1M iterations)\n");
    
    bool saw_zeros = false;
    bool saw_values = false;
    for (int i = 0; i < 32; i++) {
        if (results[i] == 0) {
            saw_zeros = true;
        } else if (results[i] > 0) {
            saw_values = true;
        }
    }
    
    if (saw_zeros && saw_values) {
        printf("\nStatus: Mixed results - some threads saw changing values (RACE!)\n");
    } else if (saw_zeros) {
        printf("\nStatus: All reads saw zeros (GPU 1 ran before GPU 0)\n");
    } else {
        printf("\nStatus: All reads saw values (timing-dependent)\n");
    }
    
    printf("\nRoot Cause: Multi-GPU read-write race (anti-dependence)\n");
    printf("            GPU 0: 32 threads write to data[0..31] (50M iterations)\n");
    printf("            GPU 1: 32 threads read from data[0..31] (1M iterations, strided)\n");
#ifdef FIXED
    printf("            FIXED: cudaStreamWaitEvent() used for proper cross-GPU sync\n");
#else
    printf("            RACE: Missing synchronization between GPU kernels\n");
    printf("            Fix: Use cudaDeviceSynchronize() or events for cross-GPU sync\n");
#endif
    
    // Cleanup
    cudaSetDevice(0);
    cudaStreamDestroy(stream0);
    
    cudaSetDevice(1);
    cudaStreamDestroy(stream1);
    
    cudaFree(data);
    cudaFree(results);
    cudaFree(dummy);
    
    printf("Test complete\n");
    
    return 0;
}

