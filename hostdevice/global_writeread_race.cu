/**
 * global_writeread_race.cu — CPU–GPU write-read race on unified memory.
 *
 * The CPU increments a counter in unified memory while a GPU kernel reads it
 * repeatedly. In race mode, the CPU starts writing while the kernel is
 * running, so the kernel can see inconsistent values (write-read race).
 * FIXED adds cudaDeviceSynchronize so the GPU finishes before the CPU writes.
 *
 * Race pattern: GPU reads unified counter while CPU writes to it.
 *
 * When FIXED is defined, host synchronizes before CPU writes (no race).
 * When FIXED is not defined, CPU writes overlap with GPU reads.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <thread>

// GPU kernel: Reads counter multiple times
__global__ void read_kernel(int* counter, int* results) {
    int tid = threadIdx.x;
    
    // Each thread reads the counter many times and accumulates
    int sum = 0;
    for (int i = 0; i < 1000000; i++) {
        int val = *counter;  // Read counter (CPU may be writing)
        sum += val;
        
        // Store sum in every iteration - compiler can't prove results[tid] doesn't alias with counter
        // This forces the read to happen in every iteration
        results[tid] = sum;
    }
}

int main() {
#ifdef FIXED
    printf("Running hostdevice/global_writeread_race (FIXED VERSION)\n");
#else
    printf("Running hostdevice/global_writeread_race (RACE VERSION)\n");
#endif
    printf("Tests write-read races when GPU reads unified memory while CPU writes to it\n\n");
    
    int* counter;
    int* results;
    
    // Allocate unified memory
    cudaMallocManaged(&counter, sizeof(int));
    cudaMallocManaged(&results, 32 * sizeof(int));
    *counter = 0;
    
    const int CPU_WRITES = 100000;
    
    // Launch GPU kernel
    read_kernel<<<1, 32>>>(counter, results);
    
#ifdef FIXED
    // FIXED: Synchronize to ensure GPU finishes reading before CPU writes
    cudaDeviceSynchronize();
#else
    // RACE: Missing synchronization! Should call cudaDeviceSynchronize() here
#endif
    
    // CPU writes to counter
#ifndef FIXED
    // ANTI-DEPENDENCE RACE: CPU immediately updates counter without waiting
    // CPU writes to counter while GPU is reading it
#endif
    // Use volatile pointer and sleep to ensure overlap and prevent optimization
    printf("CPU updating counter (GPU reading simultaneously)...\n");
    volatile int* volatile_counter = counter;  // Make counter volatile on host side
    for (int i = 0; i < CPU_WRITES; i++) {
        *volatile_counter = i + 1;  // CPU writes to volatile pointer
        std::this_thread::sleep_for(std::chrono::nanoseconds(1));  // Small sleep to ensure overlap
    }
    
    // Synchronize to ensure all GPU operations complete
    cudaDeviceSynchronize();
    
    
    
    // Cleanup
    cudaFree(counter);
    cudaFree(results);
    
    printf("Test complete\n");
    
    return 0;
}

