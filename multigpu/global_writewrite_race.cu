/**
 * global_writewrite_race.cu — Multi-GPU write-write race on unified memory.
 *
 * Two GPUs each run a kernel that writes to the same unified-memory buffer.
 * In race mode, both write to data[0], causing a write-write race across
 * GPUs. FIXED has GPU 1 write to data[1] so there is no overlap. Requires
 * at least two GPUs.
 *
 * Race pattern: GPU 0 and GPU 1 both write to the same unified memory address.
 *
 * When FIXED is defined, GPU 1 writes to data[1] (no race).
 * When FIXED is not defined, both GPUs write to data[0].
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel that writes to global memory (GPU 0 pattern)
// Single thread to avoid intra-kernel races
__global__ void writer_kernel_gpu0(int* data, int* dummy) {
    int value = 0;
    
    // Keep computing and writing frequently to increase race window
    // Single thread writes to data[0] repeatedly (2M iterations for reasonable runtime)
    for (int i = 0; i < 2000000; i++) {
        value = (value + i) % 1000000;
        data[0] = value;
        
        // Store to dummy - compiler can't prove it doesn't alias with data
        // This forces the write to happen in every iteration
        *dummy = i;
    }
    // Final write
    data[0] = value % 10;
}

// Kernel that writes to global memory (GPU 1 pattern)
// Single thread to avoid intra-kernel races
__global__ void writer_kernel_gpu1(int* data, int* dummy) {
#ifdef FIXED
    // FIXED: Write to data[1] to avoid conflict with GPU 0
    int offset = 1;
#else
    // RACE: Both GPUs write to data[0], creating write-write conflict
    int offset = 0;
#endif
    int value = 1000;
    
    // Keep computing and writing frequently to increase race window
    // Single thread writes to data[offset] repeatedly (2M iterations for reasonable runtime)
    for (int i = 0; i < 2000000; i++) {
        value = (value + i) % 1000000;
        data[offset] = value;
        
        // Store to dummy - compiler can't prove it doesn't alias with data
        // This forces the write to happen in every iteration
        *dummy = i;
    }
    // Final write
    data[offset] = value % 10;
}

int main() {
#ifdef FIXED
    printf("Running multigpu/global_writewrite_race (FIXED VERSION)\n");
#else
    printf("Running multigpu/global_writewrite_race (RACE VERSION)\n");
#endif
    printf("Tests write-write races when multiple GPUs write to same unified memory addresses\n\n");
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount < 2) {
        printf("This test requires at least 2 GPUs, found %d\n", deviceCount);
        return 1;
    }
    
    // Enable peer access between GPUs
    cudaSetDevice(0);
    cudaError_t err = cudaDeviceEnablePeerAccess(1, 0);
    
    cudaSetDevice(1);
    err = cudaDeviceEnablePeerAccess(0, 0);
    if (err == cudaSuccess) {
        printf("Enabled peer access: GPU 1 -> GPU 0\n");
    } else if (err != cudaErrorPeerAccessAlreadyEnabled) {
        printf("Warning: Could not enable peer access GPU 1 -> GPU 0: %s\n", cudaGetErrorString(err));
    }
    printf("\n");
    
    const int DATA_SIZE = 256;
    
    // Allocate unified memory accessible by all GPUs
    int* data;
    int* dummy0;
    int* dummy1;
    cudaMallocManaged(&data, DATA_SIZE * sizeof(int));
    cudaMallocManaged(&dummy0, sizeof(int));
    cudaMallocManaged(&dummy1, sizeof(int));
    
    // Initialize data
    for (int i = 0; i < DATA_SIZE; i++) {
        data[i] = -1;
    }
    *dummy0 = 0;
    *dummy1 = 0;
    
    // Create non-blocking streams for each GPU to enable concurrent execution
    cudaStream_t stream0, stream1;
    
    cudaSetDevice(0);
    cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking);
    
    cudaSetDevice(1);
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    
    // Force module loading before launching kernels to avoid serialization
    cudaFuncAttributes attr;
    cudaSetDevice(0);
    cudaFuncGetAttributes(&attr, (const void*)writer_kernel_gpu0);
    cudaSetDevice(1);
    cudaFuncGetAttributes(&attr, (const void*)writer_kernel_gpu1);
    
    printf("Launching writer kernels on both GPUs (single thread each)...\n");
    
    // Launch writer kernel on GPU 0 - single thread
    // Each GPU uses its own dummy pointer to avoid race on dummy variable
    cudaSetDevice(0);
    writer_kernel_gpu0<<<1, 1, 0, stream0>>>(data, dummy0);
    
#ifdef FIXED
    // FIXED: Properly synchronize using event to ensure GPU 1 waits for GPU 0
    cudaEvent_t sync_event;
    cudaSetDevice(0);
    cudaEventCreate(&sync_event);
    cudaEventRecord(sync_event, stream0);  // Record when GPU 0 write completes
    
    cudaSetDevice(1);
    cudaStreamWaitEvent(stream1, sync_event, 0);  // GPU 1 waits for GPU 0
    writer_kernel_gpu1<<<1, 1, 0, stream1>>>(data, dummy1);
    
    cudaEventDestroy(sync_event);
#else
    // RACE: Launch writer kernel on GPU 1 - MISSING SYNCHRONIZATION!
    // Both GPUs writing to same memory concurrently
    cudaSetDevice(1);
    writer_kernel_gpu1<<<1, 1, 0, stream1>>>(data, dummy1);
#endif
    
    // Wait for both GPUs to complete
    cudaSetDevice(0);
    cudaStreamSynchronize(stream0);
    
    cudaSetDevice(1);
    cudaStreamSynchronize(stream1);
    
    printf("Execution completed.\n\n");
    
    // Check for errors
    cudaError_t err0, err1;
    cudaSetDevice(0);
    err0 = cudaGetLastError();
    
    cudaSetDevice(1);
    err1 = cudaGetLastError();
    
    if (err0 != cudaSuccess) {
        printf("GPU 0 Error: %s\n", cudaGetErrorString(err0));
    }
    if (err1 != cudaSuccess) {
        printf("GPU 1 Error: %s\n", cudaGetErrorString(err1));
    }
    
    // Analyze results - check which GPU won
    printf("\n=== Final Results ===\n");
    printf("data[0] = %d\n", data[0]);
#ifdef FIXED
    printf("data[1] = %d\n\n", data[1]);
#else
    printf("\n");
#endif
    
    const char* status;
    if (data[0] >= 0 && data[0] <= 9) {
        status = "Valid (GPU 0 or GPU 1, both write 0-9)";
    } else {
        status = "Unexpected value!";
    }
    
    printf("Status at data[0]: %s\n", status);
    
    printf("\nStatus: WRITE-WRITE RACE EXISTS\n");
#ifdef FIXED
    printf("        GPU 0 writes to data[0], GPU 1 writes to data[1]\n");
    printf("        No race - different locations\n");
#else
    printf("        Both GPUs write to data[0] concurrently\n");
    printf("        Final value is non-deterministic\n");
#endif
    
    printf("\nRoot Cause: Multi-GPU write-write race (output dependence)\n");
    printf("            GPU 0: single thread writes to data[0] (50M iterations)\n");
#ifdef FIXED
    printf("            GPU 1: single thread writes to data[1] (50M iterations)\n");
    printf("            FIXED: cudaStreamWaitEvent() + non-overlapping locations\n");
#else
    printf("            GPU 1: single thread writes to data[0] (2M iterations)\n");
    printf("            RACE: Missing synchronization + overlapping location\n");
    printf("            Fix: Use cudaDeviceSynchronize() or events for cross-GPU sync\n");
#endif
    
    // Cleanup
    cudaSetDevice(0);
    cudaStreamDestroy(stream0);
    
    cudaSetDevice(1);
    cudaStreamDestroy(stream1);
    
    cudaFree(data);
    cudaFree(dummy0);
    cudaFree(dummy1);
    
    printf("Test complete\n");
    
    return 0;
}

