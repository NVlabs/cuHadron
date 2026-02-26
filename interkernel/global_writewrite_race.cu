/**
 * global_writewrite_race.cu — Inter-kernel global write-write race (two streams).
 *
 * Two kernels run concurrently on different streams, each writing to global
 * memory. In race mode, both write to global_data[0..31], causing a
 * write-write race across kernels. In FIXED mode, kernel 2 writes to
 * global_data[32..63] so the regions do not overlap.
 *
 * Race pattern: Kernel 1 and kernel 2 both write to global_data[0..31].
 *
 * When FIXED is defined, kernel 2 writes to [32..63] (no race).
 * When FIXED is not defined, both kernels write to [0..31].
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel 1: Writes values (single thread)
__global__ void write_kernel_1(int* global_data, int* dummy) {
    int value = 0;  // Initial value for kernel 1
    
    // Keep computing and writing frequently to increase race window
    // Single thread writes to multiple locations with strided access
    // No intra-kernel races (only 1 thread)
    for (int i = 0; i < 50000000; i++) {
        // Strided write - cycle through locations 0-31
        int idx = i % 32;
        value = (value + 1) % 1000000;
        global_data[idx] = value;
        
        // Store to dummy - compiler can't prove it doesn't alias with global_data
        // This forces the write to happen in every iteration
        *dummy = i;
    }
}

// Kernel 2: Writes values (single thread)
__global__ void write_kernel_2(int* global_data, int* dummy) {
    int value = 1000;  // Different initial value for kernel 2
    
    // Keep computing and writing frequently to increase race window
    // Single thread writes to multiple locations with strided access
    // No intra-kernel races (only 1 thread)
    for (int i = 0; i < 50000000; i++) {
        // Strided write - cycle through locations
        int idx = i % 32;
        value = (value + 1) % 1000000;
#ifdef FIXED
        // FIXED: Write to [32..63] to avoid conflict with kernel 1
        global_data[32 + idx] = value + 10000000;  // Add 10M offset
#else
        // RACE: Write to [0..31] same as kernel 1, creating write-write conflict
        global_data[idx] = value + 10000000;  // Add 10M offset
#endif
        
        // Store to dummy - compiler can't prove it doesn't alias with global_data
        // This forces the write to happen in every iteration
        *dummy = i;
    }
}

int main() {
#ifdef FIXED
    printf("Running interkernel/global_writewrite_race (FIXED VERSION)\n");
#else
    printf("Running interkernel/global_writewrite_race (RACE VERSION)\n");
#endif
    printf("Tests write-write races when two kernels on different streams write to same global memory addresses\n\n");
    
    const int N = 64;
    int h_data[N];
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = -1;
    }
    
    // Allocate device memory
    int* d_data;
    int* d_dummy1;
    int* d_dummy2;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_dummy1, sizeof(int));
    cudaMalloc(&d_dummy2, sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Create two non-blocking streams
    cudaStream_t stream1, stream2;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    
    // Force module loading
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)write_kernel_1);
    cudaFuncGetAttributes(&attr, (const void*)write_kernel_2);
    
    // Launch both kernels concurrently on different streams
    write_kernel_1<<<1, 1, 0, stream1>>>(d_data, d_dummy1);
    write_kernel_2<<<1, 1, 0, stream2>>>(d_data, d_dummy2);
    
    // Synchronize both streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // Copy results back
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data);
    cudaFree(d_dummy1);
    cudaFree(d_dummy2);
    
    printf("Test complete\n");
    
    return 0;
}

