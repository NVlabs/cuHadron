/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Inter-wave Race Test - Global Memory WAW Race
 *
 * This test creates a write-after-write (WAW) race condition in global memory
 * between blocks executing in different waves. The last thread in the grid 
 * (typically in the last wave) writes to the same memory location as the first 
 * thread (in the first wave), creating a race across waves.
 *
 * Race pattern: Thread 0 (first block/wave) and Thread (N-1) (last block/wave)
 *               both write to C[0]
 * 
 * When FIXED is defined, each thread writes to its own index (no race).
 * When FIXED is not defined, the last thread writes to index 0, creating a race.
 */

 #include <stdio.h>


 // For the CUDA runtime routines (prefixed with "cuda_")
 #include <cuda_runtime.h>
 
 
 /**
  * CUDA Kernel Device code
  *
  * Computes the vector addition of A and B into C. The 3 vectors have the same
  * number of elements numElements.
  */
__global__ void kernel(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

#ifdef FIXED
    // FIXED: No race - each thread writes to its own index
    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }
#else
    // RACE: Last thread writes to index 0, creating a race with thread 0
    int ix = i == (numElements-1) ? 0 : i;
    if (i < numElements) {
        C[ix] = A[i] + B[i] + 0.0f;
    }
#endif
}
 
 /**
  * Host main routine
  */
int main(void) {
#ifdef FIXED
  printf("Running interwave/global_writewrite_race (FIXED VERSION)\n");
#else
  printf("Running interwave/global_writewrite_race (RACE VERSION)\n");
#endif
  printf("Tests inter-wave write-write races when blocks in different waves write to same global memory\n\n");
  
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the vector length to be used, and compute its size
  int numElements = 1024 * 1024;

  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);
 
   // Allocate the host input vector A
   float *h_A = (float *)malloc(size);
 
   // Allocate the host input vector B
   float *h_B = (float *)malloc(size);
 
   // Allocate the host output vector C
   float *h_C = (float *)malloc(size);
 
   // Verify that allocations succeeded
   if (h_A == NULL || h_B == NULL || h_C == NULL) {
     fprintf(stderr, "Failed to allocate host vectors!\n");
     exit(EXIT_FAILURE);
   }
 
   // Initialize the host input vectors
   for (int i = 0; i < numElements; ++i) {
     h_A[i] = rand() / (float)RAND_MAX;
     h_B[i] = rand() / (float)RAND_MAX;
   }
 
   // Allocate the device input vector A
   float *d_A = NULL;
   err = cudaMalloc((void **)&d_A, size);
 
   if (err != cudaSuccess) {
     fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
             cudaGetErrorString(err));
     exit(EXIT_FAILURE);
   }
 
   // Allocate the device input vector B
   float *d_B = NULL;
   err = cudaMalloc((void **)&d_B, size);
 
   if (err != cudaSuccess) {
     fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
             cudaGetErrorString(err));
     exit(EXIT_FAILURE);
   }
 
   // Allocate the device output vector C
   float *d_C = NULL;
   err = cudaMalloc((void **)&d_C, size);
 
   if (err != cudaSuccess) {
     fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
             cudaGetErrorString(err));
     exit(EXIT_FAILURE);
   }
 
   // Copy the host input vectors A and B in host memory to the device input
   // vectors in
   // device memory
   printf("Copy input data from the host memory to the CUDA device\n");
   err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
 
   if (err != cudaSuccess) {
     fprintf(stderr,
             "Failed to copy vector A from host to device (error code %s)!\n",
             cudaGetErrorString(err));
     exit(EXIT_FAILURE);
   }
 
   err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
 
   if (err != cudaSuccess) {
     fprintf(stderr,
             "Failed to copy vector B from host to device (error code %s)!\n",
             cudaGetErrorString(err));
     exit(EXIT_FAILURE);
   }
 
   // Launch the Vector Add CUDA Kernel
   int threadsPerBlock = 256;
   int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
   printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
          threadsPerBlock);
   kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
   err = cudaGetLastError();
 
   if (err != cudaSuccess) {
     fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
             cudaGetErrorString(err));
     exit(EXIT_FAILURE);
   }
 
   // Copy the device result vector in device memory to the host result vector
   // in host memory.
   printf("Copy output data from the CUDA device to the host memory\n");
   err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
 
   if (err != cudaSuccess) {
     fprintf(stderr,
             "Failed to copy vector C from device to host (error code %s)!\n",
             cudaGetErrorString(err));
     exit(EXIT_FAILURE);
   }
 
    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i) {
      if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
#ifdef FIXED
          // In FIXED mode, all results should be correct
          fprintf(stderr, "Result verification failed at element %d!\n", i);
          exit(EXIT_FAILURE);
#else
          // In race mode, expect failures at element 0 (overwritten by last thread)
          if (i == 0 || i == (numElements-1)) {
              fprintf(stderr, "Result verification failed at element %d, which is expected due to race!\n", i);
          } else {
              fprintf(stderr, "Result verification failed unexpectedly at element %d!\n", i);
              exit(EXIT_FAILURE);
          }
#endif
      }
    }
 
 
   // Free device global memory
   err = cudaFree(d_A);
 
   if (err != cudaSuccess) {
     fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
             cudaGetErrorString(err));
     exit(EXIT_FAILURE);
   }
 
   err = cudaFree(d_B);
 
   if (err != cudaSuccess) {
     fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
             cudaGetErrorString(err));
     exit(EXIT_FAILURE);
   }
 
   err = cudaFree(d_C);
 
   if (err != cudaSuccess) {
     fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
             cudaGetErrorString(err));
     exit(EXIT_FAILURE);
   }
 
   // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Test complete\n");
    return 0;
}
 