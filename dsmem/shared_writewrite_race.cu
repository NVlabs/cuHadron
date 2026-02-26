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
 * shared_writewrite_race.cu — Distributed shared memory (DSMEM) write-write race.
 *
 * Two CTAs in a cluster: CTA 0 and CTA 1 both write to CTA 0’s shared memory
 * via cluster.map_shared_rank. Without cluster sync, both write concurrently
 * (inter-CTA write-write race). FIXED adds cluster.sync() so CTA 0 finishes
 * before CTA 1 writes. Requires compute capability 9.0+ (thread block clusters).
 *
 * Race pattern: CTA 0 and CTA 1 both write to CTA 0’s shared memory via DSMEM.
 *
 * When FIXED is defined, cluster.sync() ensures CTA 0 completes first (no race).
 * When FIXED is not defined, both CTAs write without synchronization.
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Kernel with distributed shared memory write-write race
// Both CTA 0 and CTA 1 write to CTA 0's shared memory
// Without proper synchronization, this creates a race
__global__ void dsmem_writewrite_kernel(int* results) {
    // Allocate shared memory accessible across the cluster
    __shared__ int shared_data[256];
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int block_rank = cluster.block_rank();
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize shared memory
    shared_data[tid] = -1;
    __syncthreads();
    // Get pointer to CTA 0's shared memory
    int* cta0_shared = cluster.map_shared_rank(shared_data, 0);
    if (block_rank == 0) {
        // CTA 0: WRITER GROUP 1 - writes to its own shared memory
        // Each thread writes to its OWN location (no intra-CTA race)
        for (int i = 0; i < 1000; i++) {
            cta0_shared[tid] = 1000 + global_tid * 10 + (i % 10);
        }
    }
#ifdef FIXED
    // FIXED: Synchronize cluster to ensure CTA 0 finishes writing before CTA 1 writes
    cluster.sync();
#endif
    if (block_rank == 1) {
        // CTA 1: WRITER GROUP 2 - also writes to CTA 0's shared memory via DSMEM
#ifndef FIXED
        // RACE: Missing cluster.sync() here! Both CTAs writing to same memory concurrently
        // INTER-CTA RACE: CTA 1 writes to CTA 0's shared memory without synchronization
#endif
        // Each thread writes to its OWN location (no intra-CTA race)
        // The race is INTER-CTA: both CTAs writing to the same shared memory locations
        for (int i = 0; i < 1000; i++) {
            cta0_shared[tid] = 2000 + global_tid * 10 + (i % 10);
        }
    }
    // Synchronize to ensure all writes complete
    cluster.sync();
    // All threads read the final values from CTA 0's shared memory
    results[global_tid] = cta0_shared[tid];
#ifdef FIXED
    // FIXED: Synchronize cluster to ensure CTA 1 finishes writing before CTA 0 exits
    cluster.sync();
#endif
}

int main() {
#ifdef FIXED
    printf("Running dsmem/shared_writewrite_race (FIXED VERSION)\n");
#else
    printf("Running dsmem/shared_writewrite_race (RACE VERSION)\n");
#endif
    printf("Tests write-write races in distributed shared memory when CTAs in cluster write to same memory\n\n");
    
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
        
        cudaLaunchKernelEx(&config, dsmem_writewrite_kernel, results);
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

