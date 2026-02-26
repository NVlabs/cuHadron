/**
 * global_bytewise_writewrite.cu — False-positive check: byte-level independence (global).
 *
 * Two threads in different warps write to different bytes of the same 4-byte
 * word in global memory (thread 0 → byte 0, thread 32 → byte 1). There is no
 * data race; bytes are independent. A detector that reports a race here
 * produces a false positive. Used to check byte-level vs word-level handling.
 *
 * Pattern: Thread 0 writes byte 0, thread 32 writes byte 1 of the same int.
 *
 * No FIXED variant; the test always has no race (byte-level independence).
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel where threads write to different bytes of the same int
__global__ void bytewise_race_kernel(int* global_data, int* results) {
    int tid = threadIdx.x;
    
    // RACE: Warp 0 thread 0 and Warp 1 thread 0 write to different bytes
    // of the same 4-byte integer in global memory
    if (tid == 0) {
        // Warp 0, thread 0: Write to byte 0
        char* byte_ptr = (char*)global_data;
        byte_ptr[0] = 0xAA;  // Write 0xAA to first byte
    }
    else if (tid == 32) {
        // Warp 1, thread 0: Write to byte 1
        char* byte_ptr = (char*)global_data;
        byte_ptr[1] = 0xBB;  // Write 0xBB to second byte
    }
    
    // Wait for both writes to complete
    __syncthreads();
    
    // Read final value
    if (tid == 0) {
        results[0] = *global_data;
    }
}

int main() {
    printf("Running false_positives/global_bytewise_writewrite (false positive check, no race)\n");
    printf("Tests byte-level granularity - different bytes in same word (false positive check)\n\n");
    
    int* global_data;
    int* results;
    
    cudaMalloc(&global_data, sizeof(int));
    cudaMallocManaged(&results, sizeof(int));
    
    // Initialize global data to zero
    int init_value = 0;
    cudaMemcpy(global_data, &init_value, sizeof(int), cudaMemcpyHostToDevice);
    
    results[0] = 0;
    
    // Launch kernel with 1 block, 64 threads (2 warps)
    bytewise_race_kernel<<<1, 64>>>(global_data, results);
    cudaDeviceSynchronize();
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("Execution completed.\n\n");
    
    // Print result
    int final_value = results[0];
    printf("Final value: 0x%08X\n", final_value);
    
    // Expected if no race: 0x0000BBAA (both bytes written)
    // Could also be 0x000000AA or 0x0000BB00 if timing causes issues
    
    unsigned char byte0 = (final_value >> 0) & 0xFF;
    unsigned char byte1 = (final_value >> 8) & 0xFF;
    unsigned char byte2 = (final_value >> 16) & 0xFF;
    unsigned char byte3 = (final_value >> 24) & 0xFF;
    
    printf("Byte 0: 0x%02X (expected 0xAA)\n", byte0);
    printf("Byte 1: 0x%02X (expected 0xBB)\n", byte1);
    printf("Byte 2: 0x%02X (expected 0x00)\n", byte2);
    printf("Byte 3: 0x%02X (expected 0x00)\n", byte3);
    printf("\n");
    
    printf("=== Analysis ===\n");
    printf("This test writes to DIFFERENT BYTES of the same word.\n");
    printf("Race detectors must determine if this is a race:\n");
    printf("  - Word-level granularity: YES, this is a race (both access same word)\n");
    printf("  - Byte-level granularity: NO, this is not a race (different bytes)\n");
    printf("\n");
    printf("Root Cause: Byte-level write-write to same word in global memory\n");
    printf("            Two threads from different warps write to different bytes\n");
    printf("            No synchronization between write operations\n");
    
    // Cleanup
    cudaFree(global_data);
    cudaFree(results);
    
    printf("Test complete\n");
    
    return 0;
}

