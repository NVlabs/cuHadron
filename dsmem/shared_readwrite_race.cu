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
 * shared_readwrite_race.cu — Distributed shared memory (DSMEM) read-write race.
 *
 * CTA 0 writes to its shared memory; CTA 1 reads that memory via
 * cluster.map_shared_rank. Without cluster sync, CTA 1 can read before CTA 0
 * has finished (inter-CTA read-write race). FIXED adds cluster.sync() so CTA 0
 * completes before CTA 1 reads. Requires compute capability 9.0+ (clusters).
 *
 * Race pattern: CTA 1 reads CTA 0’s shared memory via DSMEM while CTA 0 writes it.
 *
 * When FIXED is defined, cluster.sync() ensures CTA 0 completes first (no race).
 * When FIXED is not defined, CTA 1 reads without synchronization.
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Kernel with distributed shared memory read-write race
// CTA 0 writes to its shared memory, CTA 1 reads from CTA 0's shared memory
// Without proper synchronization, this creates a race
__global__ void dsmem_readwrite_kernel(int* results) {
    // Allocate shared memory accessible across the cluster
    __shared__ int shared_data[256];
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int block_rank = cluster.block_rank();
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize shared memory
    shared_data[tid] = 0;
    __syncthreads();
    if (block_rank == 0) {
        // CTA 0: WRITER - writes to its own shared memory
        // Each thread writes to its OWN location repeatedly (no intra-CTA race)
        // Unpredictable break prevents compiler from collapsing to single write
        for (int i = 0; i < 1000; i++) {
            int val = global_tid * 100 + i;
            shared_data[tid] = val;
            // Unpredictable break - compiler can't optimize loop away
            if (val < 0 || val > 1000000) break;
        }
        // Store final value written
        results[global_tid] = shared_data[tid];
    }
#ifdef FIXED
    // FIXED: Synchronize cluster to ensure CTA 0 finishes writing before CTA 1 reads
    cluster.sync();
#endif
    if (block_rank == 1) {
        // CTA 1: READER - reads from CTA 0's shared memory via DSMEM
#ifndef FIXED
        // RACE: Missing cluster.sync() here! Should wait for CTA 0 to finish writing
        // INTER-CTA RACE: CTA 1 reads CTA 0's shared memory without synchronization
#endif
        // Access CTA 0's shared memory using distributed shared memory
        int* cta0_shared = cluster.map_shared_rank(shared_data, 0);
        // Each thread reads from its OWN location (no intra-CTA race)
        // The race is INTER-CTA: reading from CTA 0's memory while CTA 0 may be writing
        long long sum = 0;
        for (int i = 0; i < 1000; i++) {
            int val = cta0_shared[tid];
            sum += val;
            // Unpredictable break - compiler can't optimize loop away
            if (val < 0 || val > 1000000) break;
        }
        results[global_tid] = (int)(sum / 1000);
    }
#ifdef FIXED
    // FIXED: Synchronize cluster to ensure CTA 1 finishes writing before CTA 0 exits
    cluster.sync();
#endif
}

int main() {
#ifdef FIXED
    printf("Running dsmem/shared_readwrite_race (FIXED VERSION)\n");
#else
    printf("Running dsmem/shared_readwrite_race (RACE VERSION)\n");
#endif
    printf("Tests read-write races in distributed shared memory when CTAs read without synchronization\n\n");
    
    int device;
    cudaGetDevice(&device);
    
    // Check if device supports clusters
    int cluster_supported = 0;
    cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch, device);
    
    if (!cluster_supported) {
        printf("Device does not support thread block clusters (compute capability 9.0+ required)\n");
        return 1;
    }
    
    const int THREADS_PER_BLOCK = 256;
    const int NUM_BLOCKS = 2;  // 2 CTAs in the cluster
    const int TOTAL_THREADS = THREADS_PER_BLOCK * NUM_BLOCKS;
    
    int* results;
    cudaMallocManaged(&results, TOTAL_THREADS * sizeof(int));
    
    // Initialize results
    for (int i = 0; i < TOTAL_THREADS; i++) {
        results[i] = -1;
    }
    
    // Launch kernel with cluster
    {
        cudaLaunchConfig_t config = {0};
        config.gridDim = NUM_BLOCKS;
        config.blockDim = THREADS_PER_BLOCK;
        
        // Configure cluster dimensions
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = NUM_BLOCKS;  // 2 blocks in cluster
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        
        config.attrs = attrs;
        config.numAttrs = 1;
        
        cudaLaunchKernelEx(&config, dsmem_readwrite_kernel, results);
    }
    
    cudaDeviceSynchronize();
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    
    // Cleanup
    cudaFree(results);
    
    printf("Test complete\n");
    
    return 0;
}

