/**
 * shared_readwrite_race.cu — Inter-subwarp shared-memory read-write race.
 *
 * Subwarp 0 writes to shared_data[0..15]; subwarp 1 reads. In race mode,
 * subwarp 1 reads from [0..15], the same region subwarp 0 writes (read-write
 * race). In FIXED mode, subwarp 1 reads from [16..31] so there is no overlap.
 *
 * Race pattern: Subwarp 0 writes shared_data[0..15]; subwarp 1 reads same region.
 *
 * When FIXED is defined, subwarp 1 reads from [16..31] (no race).
 * When FIXED is not defined, subwarp 1 reads from [0..15] (race).
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel with divergent threads causing read-write race on shared memory
__global__ void divergent_readwrite_kernel(int* results) {
    __shared__ int shared_data[32];
    
    int tid = threadIdx.x;
    int subwarp_id = tid / 16;  // Subwarp 0: threads 0-15, Subwarp 1: threads 16-31
    int lane_in_subwarp = tid % 16;
    
    // Initialize shared memory
    shared_data[tid] = 0;
    __syncthreads();  // Safe initialization
    
    // DIVERGENT BRANCH: Subwarp 0 WRITES, subwarp 1 READS from shared memory
    // Subwarp 0: Write to shared_data[0..15]
    // Subwarp 1: Read from shared_data[0..15] (RACE) or [16..31] (FIXED)
    // This creates INTER-SUBWARP races but NO intra-subwarp races
    
    if (subwarp_id == 0) {
        // SUBWARP 0: Write to shared_data[0..15]
        for (int i = 0; i < 10000; i++) {
            shared_data[lane_in_subwarp] = tid + i;
        }
        // Store final written value
        results[tid] = shared_data[lane_in_subwarp];
    } else if (subwarp_id == 1) {
        // SUBWARP 1: Read from shared_data
#ifdef FIXED
        int offset = 16;  // FIXED: Read from different region [16..31]
#else
        int offset = 0;  // RACE: Read from same region [0..15] as subwarp 0 writes
#endif
        int sum = 0;
        for (int i = 0; i < 10000; i++) {
            int val = shared_data[offset + lane_in_subwarp];
            sum += val;
            // Write to a separate region - compiler can't prove this doesn't alias with read
            // This forces the read to happen in every iteration without creating intra-subwarp conflicts
            shared_data[tid] = i;  // tid is [16..31] for subwarp 1, writes to [16..31]
        }
        results[tid] = sum;
    }
}

int main() {
#ifdef FIXED
    printf("Running intersubwarp/shared_readwrite_race (FIXED VERSION)\n");
#else
    printf("Running intersubwarp/shared_readwrite_race (RACE VERSION)\n");
#endif
    printf("Tests read-write races between divergent threads in same warp accessing shared memory\n\n");
    
    int* results;
    const int N = 32;
    
    // Allocate memory for results
    cudaMallocManaged(&results, N * sizeof(int));
    
    // Initialize results
    for (int i = 0; i < N; i++) {
        results[i] = -1;
    }
#ifdef FIXED
    printf("FIXED: Subwarp 0 writes to [0..15], Subwarp 1 reads from [16..31] (no overlap)\n");
    printf("  Subwarp 0 (threads 0-15): WRITE to shared_data[0..15]\n");
    printf("  Subwarp 1 (threads 16-31): READ from shared_data[16..31]\n");
#else
    printf("RACE: Subwarp 0 writes to [0..15], Subwarp 1 reads from [0..15] (conflict!)\n");
    printf("  Subwarp 0 (threads 0-15): WRITE to shared_data[0..15]\n");
    printf("  Subwarp 1 (threads 16-31): READ from shared_data[0..15] (concurrent!)\n");
#endif
    printf("No __syncthreads() between subwarp operations!\n\n");
    
    // Launch kernel with 1 block, 32 threads (1 warp)
    divergent_readwrite_kernel<<<1, 32>>>(results);
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

