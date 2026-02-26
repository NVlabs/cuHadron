/**
 * global_readwrite_race.cu — cp.async.bulk shared-to-global read-write race.
 *
 * Thread 0 uses cp.async.bulk to copy from shared to global. Other threads
 * read from global. In race mode, they read before the bulk copy completes,
 * so they can see old values (race). FIXED adds wait_group and __syncthreads
 * before other threads read global.
 *
 * Race pattern: Threads 1–63 read global_data before bulk copy completes.
 *
 * When FIXED is defined, wait_group and __syncthreads before reads.
 * When FIXED is not defined, other threads read global without waiting.
 */

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void async_bulk_race_kernel(int* global_data, int* result) {
    __shared__ int shared_data[64];
    
    int tid = threadIdx.x;
    
    // Initialize shared memory with known values
    shared_data[tid] = tid * 10;
    __syncthreads();
    
    // Thread 0 uses cp.async.bulk to copy from shared to global memory
    if (tid == 0) {
        // Bulk copy 256 bytes (64 ints) from shared[0] to global[0]
        unsigned int shared_addr = __cvta_generic_to_shared(&shared_data[0]);
        asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 256;\n" 
                     :: "l"(&global_data[0]), 
                        "r"(shared_addr));
        asm volatile("cp.async.bulk.commit_group;");
#ifdef FIXED
        // FIXED: Wait for bulk copy to complete before other threads read
        asm volatile("cp.async.bulk.wait_group 0;\n");
#endif
    }
    
#ifdef FIXED
    // FIXED: Synchronize so all threads wait for bulk copy
    __syncthreads();
    int value = global_data[tid];
#else
    // RACE: Missing synchronization! Should call cp.async.bulk.wait_group here
    // Race: cp.async.bulk might not have completed yet when threads read
    
    // All threads immediately try to read from global memory
    // This creates a race! Threads might read the old value or the new copied value
    int value = global_data[tid];
    
    // Now wait (too late!) 
    if (tid == 0) {
        asm volatile("cp.async.bulk.wait_group 0;\n");
    }
    __syncthreads();
#endif
    
    // Store what we read (racing value) and what the correct value should be
    result[tid] = value;                    // What we actually read (race)
    result[tid + 64] = global_data[tid];    // What it should be (after wait)
}

int main() {
#ifdef FIXED
    printf("Running bulkcpy/global_readwrite_race (FIXED VERSION)\n");
#else
    printf("Running bulkcpy/global_readwrite_race (RACE VERSION)\n");
#endif
    printf("Tests read-write races when threads read from global memory before bulk copy completes\n\n");
    
    const int N = 64;
    int h_data[N];
    int h_result[N * 2];
    
    // Initialize global data
    for (int i = 0; i < N; i++) {
        h_data[i] = -999;
    }
    
    // Allocate device memory
    int* d_data;
    int* d_result;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_result, N * 2 * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    async_bulk_race_kernel<<<1, 64>>>(d_data, d_result);
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_result, d_result, N * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_result);
    
    printf("Test complete\n");
    
    return 0;
}

