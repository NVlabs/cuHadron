/**
 * shared_writewrite_race.cu — Inter-subwarp shared-memory write-write race.
 *
 * Two subwarps (threads 0–15 and 16–31) take divergent paths; both write to
 * shared memory. In race mode, subwarp 1 writes to the same region [0..15] as
 * subwarp 0, causing an inter-subwarp write-write race. In FIXED mode, subwarp 1
 * writes to [16..31] so there is no overlap.
 *
 * Race pattern: Subwarp 0 and subwarp 1 both write to shared_data[0..15].
 *
 * When FIXED is defined, subwarp 1 writes to [16..31] (no race).
 * When FIXED is not defined, subwarp 1 writes to [0..15] (race).
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel with divergent threads causing write-write race on shared memory
__global__ void divergent_writewrite_kernel(int* results) {
    __shared__ int shared_data[32];
    
    int tid = threadIdx.x;
    int subwarp_id = tid / 16;  // Subwarp 0: threads 0-15, Subwarp 1: threads 16-31
    int lane_in_subwarp = tid % 16;
    
    // Initialize shared memory
    shared_data[tid] = 0;
    __syncthreads();  // Safe initialization
    
    // DIVERGENT BRANCH: Subwarps 0 and 1 write to shared memory
    // This creates INTER-SUBWARP races but NO intra-subwarp races
    
    if (subwarp_id == 0) {
        // SUBWARP 0: Write to shared_data[0..15]
        for (int i = 0; i < 10000; i++) {
            int value = 1000 + tid * 10 + (i % 10);
            shared_data[lane_in_subwarp] = value;
        }
    } else if (subwarp_id == 1) {
        // SUBWARP 1: Write to shared_data
#ifdef FIXED
        // FIXED: Write to different region [16..31] to avoid race
        int offset = 16;
#else
        // RACE: Write to same region [0..15] as subwarp 0!
        int offset = 0;
#endif
        for (int i = 0; i < 10000; i++) {
            int value = 2000 + tid * 10 + (i % 10);
            shared_data[offset + lane_in_subwarp] = value;
        }
    }
    
    // All threads read the final values (to see which group won)
    __syncthreads();  // Sync here to ensure writes are done
    // Read from one of the contested locations
    results[tid] = shared_data[0];
}

int main() {
#ifdef FIXED
    printf("Running intersubwarp/shared_writewrite_race (FIXED VERSION)\n");
#else
    printf("Running intersubwarp/shared_writewrite_race (RACE VERSION)\n");
#endif
    printf("Tests write-write races between divergent threads in same warp writing to shared memory\n\n");
    
    int* results;
    const int N = 32;
    
    // Allocate memory for results
    cudaMallocManaged(&results, N * sizeof(int));
    
    // Initialize results
    for (int i = 0; i < N; i++) {
        results[i] = -1;
    }
#ifdef FIXED
    printf("FIXED: Each subwarp writes to its own region (no overlap)\n");
    printf("  Subwarp 0 (threads 0-15): WRITE to shared_data[0..15]\n");
    printf("  Subwarp 1 (threads 16-31): WRITE to shared_data[16..31]\n");
#else
    printf("RACE: Both subwarps write to shared_data[0..15]\n");
    printf("  Subwarp 0 (threads 0-15): WRITE to shared_data[0..15]\n");
    printf("  Subwarp 1 (threads 16-31): WRITE to shared_data[0..15] (conflict!)\n");
#endif
    printf("No __syncthreads() between subwarp writes!\n\n");
    
    // Launch kernel with 1 block, 32 threads (1 warp)
    divergent_writewrite_kernel<<<1, 32>>>(results);
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

