'''
shared memory 배열을 선언한다
global memory에서 shared memory로 직접 복사한다
k는 shared memory를 넘어서지 않기 위해
'''
import torch
from torch.utils.cpp_extension import load_inline

TILE = 16

cuda_src = f"""
#include <torch/extension.h>

#define TILE {TILE}

__global__ void matmul_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {{
        sum += A[row * K + k] * B[k * N + col];
    }}
    C[row * N + col] = sum;
}}

__global__ void matmul_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    int tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < tiles; t++) {{
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {{
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }}

        __syncthreads();
    }}

    if (row < M && col < N)
        C[row * N + col] = sum;
}}

void launch_naive(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                  int M, int N, int K) {{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_naive<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K);
}}

void launch_tiled(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                  int M, int N, int K) {{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K);
}}
"""

cpp_src = """
void launch_naive(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                  int M, int N, int K);
void launch_tiled(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                  int M, int N, int K);
"""

module = load_inline(
    name="matmul_compare",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["launch_naive", "launch_tiled"],
    verbose=False,
)

def matmul_naive(A, B):
    M, K = A.shape
    _, N = B.shape
    C = torch.zeros((M, N), device="cuda")
    module.launch_naive(A, B, C, M, N, K)
    return C

def matmul_tiled(A, B):
    M, K = A.shape
    _, N = B.shape
    C = torch.zeros((M, N), device="cuda")
    module.launch_tiled(A, B, C, M, N, K)
    return C

def benchmark(fn, A, B, iters=20):
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    start.record()
    for _ in range(iters):
        fn(A, B)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

if __name__ == "__main__":
    M = N = K = 1024

    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")

    ref = torch.mm(A, B)
    out_naive = matmul_naive(A, B)
    out_tiled = matmul_tiled(A, B)

    print("error naive:", (ref - out_naive).abs().max().item())
    print("error tiled:", (ref - out_tiled).abs().max().item())

    t_naive = benchmark(matmul_naive, A, B)
    t_tiled = benchmark(matmul_tiled, A, B)
    t_torch = benchmark(torch.mm, A, B)

    print()
    print("naive  ms:", round(t_naive, 3))
    print("tiled  ms:", round(t_tiled, 3))
    print("torch  ms:", round(t_torch, 3))
    print("speedup (tiled / naive):", round(t_naive / t_tiled, 2))
