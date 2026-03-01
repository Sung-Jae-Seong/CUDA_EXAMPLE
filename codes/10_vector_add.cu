// ncu 분석을 위한 예제 파일
// nvcc -arch=native 10_vector_add.cu -o my_cuda_program
// sudo /usr/local/cuda-12.4/bin/ncu --launch-skip 10 --launch-count 1 ./my_cuda_program
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

inline void check(cudaError_t e) {
    if (e != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(e));
        exit(1);
    }
}

__global__
void vectorAdd(float* a, float* b, float* c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 24;   // 16M elements
    size_t size = N * sizeof(float);

    float *a, *b, *c;
    check(cudaMallocManaged(&a, size));
    check(cudaMallocManaged(&b, size));
    check(cudaMallocManaged(&c, size));

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    int deviceId;
    cudaDeviceProp props;
    check(cudaGetDevice(&deviceId));
    check(cudaGetDeviceProperties(&props, deviceId));

    int threads = 256;
    int blocks = props.multiProcessorCount * 4;

    // 커널 반복 실행 → 프로파일링 안정화
    for (int it = 0; it < 100; it++) {
        vectorAdd<<<blocks, threads>>>(a, b, c, N);
    }

    check(cudaDeviceSynchronize());

    printf("c[0] = %f\n", c[0]);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}
