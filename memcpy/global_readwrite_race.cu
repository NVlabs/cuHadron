/**
 * global_readwrite_race.cu — cp.async global-memory read-write race.
 *
 * Warp 0 uses cp.async to read from global_data into shared memory. In race
 * mode, warp 1 may write to the same global indices without synchronization,
 * so the async copy can read old or new values (race). In FIXED mode, warp 1
 * writes and __syncthreads() ensures ordering before warp 0’s copy.
 *
 * Race pattern: cp.async reads global_data[0..31] while other threads may write it.
 *
 * When FIXED is defined, warp 1 writes and syncthreads precedes warp 0’s copy.
 * When FIXED is not defined, no sync; cp.async can race with warp 1 writes.
 */

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void global_race_kernel(int* global_data) {
    __shared__ int shared_data[64];
    
    int tid = threadIdx.x;
    
    // Compute global memory index: each warp maps to indices 0-31
    int global_idx = tid % 32;
      
    // Threads in warp 0 (0-31): Use cp.async to READ from global_data[tid % 32]
#ifndef FIXED
    // RACE: cp.async might read old or new value
#else
    // FIXED: Ensure warp 1 writes complete before warp 0 reads
    if (tid >= 32) {
        global_data[global_idx] = 900 + tid;
    }
    __syncthreads();
#endif
    
    if (tid < 32) {
        unsigned int shared_addr = __cvta_generic_to_shared(&shared_data[tid]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" 
                     :: "r"(shared_addr), 
                        "l"(&global_data[global_idx]));
    
        // Wait for all async copies to complete
        asm volatile("cp.async.wait_all;\n");
        
        // Copy back from shared to global so we can see what was read
        global_data[tid + 32] = shared_data[tid];
    }

#ifndef FIXED
    // RACE: Threads in warp 1 (32-63): WRITE to global_data[tid % 32] concurrently
    if (tid >= 32) {
        global_data[global_idx] = 900 + tid;
    }
#endif
}

int main() {
#ifdef FIXED
    printf("Running memcpy/global_readwrite_race (FIXED VERSION)\n");
#else
    printf("Running memcpy/global_readwrite_race (RACE VERSION)\n");
#endif
    printf("Tests read-write races in global memory when one warp reads via cp.async while another writes\n\n");
    
    const int N = 64;
    int h_data[N];
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 10;
    }
    
    // Allocate device memory
    int* d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    global_race_kernel<<<1, 64>>>(d_data);
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_data);
    
    printf("Test complete\n");
    
    return 0;
}

