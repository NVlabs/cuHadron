/**
 * shared_writeread_race_g2s.cu — cp.async.bulk global-to-shared write-read race.
 *
 * Thread 0 uses cp.async.bulk (with mbarrier) to copy global to shared.
 * Threads 32–63 read from the same shared region. In race mode, they read
 * before the bulk copy completes (write-read race). FIXED adds
 * mbarrier.wait before the readers access shared.
 *
 * Race pattern: Threads 32–63 read shared[0..63] while thread 0’s bulk copy writes it.
 *
 * When FIXED is defined, mbarrier wait ensures copy completes before reads.
 * When FIXED is not defined, readers access shared without waiting.
 */

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void shared_bulk_race_kernel(int* global_data) {
    __shared__ int shared_data[64];
    __shared__ unsigned long long mbar;
    unsigned int mbar_addr = __cvta_generic_to_shared(&mbar);
   
    int tid = threadIdx.x;
    
    // Initialize shared memory with initial values
    for (int i = 0; i < 64; i++) {
        if (tid == 0) shared_data[i] = -1;
    }
    
    // Initialize mbarrier
    if (tid == 0) {
        unsigned int mbar_addr = __cvta_generic_to_shared(&mbar);
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mbar_addr));
        // Expect 256 bytes for the bulk copy
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], 256;" :: "r"(mbar_addr));
    }
    __syncthreads();
    
    // RACE: Thread 0 writes to shared memory (via bulk copy) while threads 32-63 read from it
    // Thread 0: Bulk copy from global[0:63] to shared[0:63] - WRITING to shared memory
    // Threads 32-63: READ from shared[tid - 32] - READING the shared memory being written!
    
    if (tid == 0) {
        // Thread 0: Bulk copy 256 bytes (64 ints) from global[0] to shared[0]
        // This WRITE to shared memory races with reads from threads 32-63
        unsigned int shared_addr = __cvta_generic_to_shared(&shared_data[0]);
        unsigned int mbar_addr = __cvta_generic_to_shared(&mbar);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 256, [%2];" 
                     :: "r"(shared_addr),
                        "l"(&global_data[0]),
                        "r"(mbar_addr));
    }
#ifdef FIXED
    // FIXED: Add synchronization to ensure bulk copy completes before reads
    unsigned long long state = 0;
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "try_wait_loop:\n\t"
                 "mbarrier.try_wait.shared.b64 p, [%0], %1;\n\t"
                 "@!p bra try_wait_loop;\n\t"
                 "}" :: "r"(mbar_addr), "l"(state));
#endif    
    // Threads 32-63: READ from shared memory that thread 0 is copying to
    int value = -999;
    if (tid >= 32 && tid < 64) {
        int shared_idx = tid - 32;
        value = shared_data[shared_idx];
    }
    
    __syncthreads();
    
    // Write back what we read (for verification)
    if (tid >= 32 && tid < 64) {
        int result_idx = tid - 32 + 64;
        global_data[result_idx] = value;
    }
}

int main() {
#ifdef FIXED
    printf("Running bulkcpy/shared_writeread_race_g2s (FIXED VERSION)\n");
#else
    printf("Running bulkcpy/shared_writeread_race_g2s (RACE VERSION)\n");
#endif
    printf("Tests write-read races in shared memory when bulk copy writes while other threads read (global to shared)\n\n");
    
    const int N = 128;  // 64 for input, 64 for results
    int h_data[N];
    
    // Initialize first 64 elements with known values
    for (int i = 0; i < 64; i++) {
        h_data[i] = i * 10;
    }
    // Initialize result area
    for (int i = 64; i < N; i++) {
        h_data[i] = -1;
    }
    
    // Allocate device memory
    int* d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    shared_bulk_race_kernel<<<1, 64>>>(d_data);
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_data);
    
    printf("Test complete\n");
    
    return 0;
}

