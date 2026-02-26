/**
 * global_writewrite_race.cu — CPU–GPU write-write race on unified memory.
 *
 * A GPU kernel and the CPU both write to the same unified-memory buffer. In
 * race mode, the CPU starts writing while the kernel is still running (or
 * without synchronizing after the kernel), so CPU and GPU writes conflict.
 * FIXED adds cudaDeviceSynchronize before the CPU writes.
 *
 * Race pattern: CPU and GPU both write to the same unified memory addresses.
 *
 * When FIXED is defined, host synchronizes before CPU writes (no race).
 * When FIXED is not defined, CPU writes overlap with GPU writes.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <thread>

// Kernel writes to global memory repeatedly
__global__ void write_kernel(int* data, int num_writes, int* dummy) {
    int tid = threadIdx.x;
    
    // Each thread writes to its OWN location to avoid intra-kernel races
    // Write changing values to prevent loop optimization
    int value = tid * 100;
    for (int i = 0; i < num_writes; i++) {
        value = (value + 1) % 10000;  // Changing value in each iteration
        data[tid] = value;  // Each thread writes to its own tid location
        
        // Each thread writes to its own dummy location to avoid intra-kernel race
        // Compiler can't prove dummy[tid] doesn't alias with data
        // This forces the write to happen in every iteration
        dummy[tid] = i;
    }
}

int main() {
#ifdef FIXED
    printf("Running hostdevice/global_writewrite_race (FIXED VERSION)\n");
#else
    printf("Running hostdevice/global_writewrite_race (RACE VERSION)\n");
#endif
    printf("Tests write-write races when CPU and GPU write to same unified memory addresses\n\n");
    
    const int N = 64;
    int* data;
    int* dummy;
    
    // Allocate unified memory accessible by both CPU and GPU
    cudaMallocManaged(&data, N * sizeof(int));
    cudaMallocManaged(&dummy, 32 * sizeof(int));  // 32 threads need 32 dummy locations
    
    // Initialize data to sentinel values
    for (int i = 0; i < N; i++) {
        data[i] = -1;
    }
    for (int i = 0; i < 32; i++) {
        dummy[i] = 0;
    }
    
    const int GPU_WRITES = 1000000;  // 1M writes per thread
    const int CPU_WRITES = 100000;  // 100K writes
    
    // Launch GPU kernel that writes repeatedly
    write_kernel<<<1, 32>>>(data, GPU_WRITES, dummy);
    
#ifdef FIXED
    // FIXED: Synchronize to ensure GPU finishes writing before CPU writes
    cudaDeviceSynchronize();
#else
    // RACE: Missing synchronization! Should call cudaDeviceSynchronize() here
#endif
    
    // CPU writes to same locations
    volatile int* volatile_data = data;
    for (int i = 0; i < CPU_WRITES; i++) {
        int idx = i % 32;
        volatile_data[idx] = 900 + idx;  // CPU writes to volatile pointer
        std::this_thread::sleep_for(std::chrono::nanoseconds(1));  // Small sleep to ensure overlap
    }
    
    // Synchronize to ensure GPU completes
    cudaDeviceSynchronize();
    
    // Final values captured
    
    
    // Cleanup
    cudaFree(data);
    cudaFree(dummy);
    
    printf("Test complete\n");
    
    return 0;
}

