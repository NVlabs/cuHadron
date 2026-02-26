/**
 * global_readwrite_race.cu — Inter-kernel global read-write race (producer/consumer).
 *
 * Producer and consumer kernels run on different streams. Producer writes
 * global_data[tid]; consumer reads global_data with strided indices. Without
 * synchronization they overlap, so the consumer can read before the producer
 * has written (read-write race). FIXED adds cudaDeviceSynchronize between
 * launch and consumer.
 *
 * Race pattern: Consumer reads global_data while producer writes it.
 *
 * When FIXED is defined, host synchronizes between producer and consumer.
 * When FIXED is not defined, both run concurrently (race).
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Producer kernel: Computes and writes results
__global__ void producer_kernel(int* global_data) {
    int tid = threadIdx.x;
    int value = tid * tid;
    
    // Keep computing and writing frequently to increase race window
    // Increase iterations to make producer run longer and overlap with consumer
    for (int i = 0; i < 50000000; i++) {
        value = (value + i) % 1000000;
        // Write to memory frequently
        global_data[tid] = value;
    }
    // Final write
    global_data[tid] = tid * tid + (value % 10);
}

// Consumer kernel: Reads producer's output
__global__ void consumer_kernel(int* global_data, int* result) {
    int tid = threadIdx.x;
    long long sum = 0;
    
    // Keep reading frequently to increase race window
    // Use strided reads to make each read unique
    for (int i = 0; i < 1000000; i++) {
        // Strided read - access different location each iteration
        int idx = (tid + i) % 32;
        sum += global_data[idx];
    }
    // Store accumulated sum (divide to fit in int range)
    result[tid] = (int)(sum / 1000);
}

int main() {
#ifdef FIXED
    printf("Running interkernel/global_readwrite_race (FIXED VERSION)\n");
#else
    printf("Running interkernel/global_readwrite_race (RACE VERSION)\n");
#endif
    printf("Tests read-write races when consumer kernel reads before producer kernel completes on different streams\n\n");
    
    const int N = 32;
    int h_data[N];
    int h_result[N];
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = -999;
    }
    
    // Allocate device memory
    int* d_data;
    int* d_result;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_result, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Create two non-blocking streams
    cudaStream_t stream1, stream2;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    
    // Force module loading
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)producer_kernel);
    cudaFuncGetAttributes(&attr, (const void*)consumer_kernel);
    
    // Launch producer on stream1
    producer_kernel<<<1, 32, 0, stream1>>>(d_data);
    
#ifdef FIXED
    // FIXED: Synchronize using cudaEvent
    cudaEvent_t sync_event;
    cudaEventCreate(&sync_event);
    cudaEventRecord(sync_event, stream1);
    cudaStreamWaitEvent(stream2, sync_event, 0);
    consumer_kernel<<<1, 32, 0, stream2>>>(d_data, d_result);
    cudaEventDestroy(sync_event);
#else
    // RACE: Missing synchronization
    consumer_kernel<<<1, 32, 0, stream2>>>(d_data, d_result);
#endif
    
    // Synchronize both streams
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result, d_result, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data);
    cudaFree(d_result);
    
    printf("Test complete\n");
    
    return 0;
}

