// Shared host driver: two blocks of 32 threads (producer = block 1, consumer = block 0), result check.
int main() {
#ifdef FIXED
    printf("Running %s (FIXED VERSION)\n", TEST_NAME);
#else
    printf("Running %s (RACE VERSION)\n", TEST_NAME);
#endif
    printf("%s\n\n", TEST_DESC);
    unsigned *d_buf;
    cudaMalloc(&d_buf, 4096 * sizeof(unsigned));
    cudaMemset(d_buf, 0, 4096 * sizeof(unsigned));
    kernel<<<2, 32>>>(d_buf);
    cudaError_t err = cudaDeviceSynchronize();
    unsigned h[64];
    cudaMemcpy(h, d_buf + 1024, sizeof h, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < 32; i++) bad += h[i] != (unsigned)(i + 1);
    printf("%s: %d of 32 values the consumer read were stale\n", err == cudaSuccess ? "Done" : cudaGetErrorString(err), bad);
    cudaFree(d_buf);
    printf("Test complete\n");
    return 0;
}
