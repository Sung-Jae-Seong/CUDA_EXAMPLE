import torch
from torch.utils.cpp_extension import load_inline

TILE = 32

cuda_src = f"""
#include <torch/extension.h>

#define TILE {TILE}

__global__ void transpose_naive(
    const float* input,
    float* output,
    int M, int N
) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {{
        output[col * M + row] = input[row * N + col];
    }}
}}

// Read + Write 모두 coalesced
__global__ void transpose_tiled(
    const float* input,
    float* output,
    int M, int N
) {{
    __shared__ float tile[TILE][TILE + 1]; // bank conflict 방지

    int in_row = blockIdx.y * TILE + threadIdx.y; // tile 크기로 쪼개기 위함
    int in_col = blockIdx.x * TILE + threadIdx.x;

    if (in_row < M && in_col < N) {{
        tile[threadIdx.y][threadIdx.x] =
            input[in_row * N + in_col];
    }}

    __syncthreads();

    int out_row = blockIdx.x * TILE + threadIdx.y;
    int out_col = blockIdx.y * TILE + threadIdx.x;

    if (out_row < N && out_col < M) {{
        output[out_row * M + out_col] =
            tile[threadIdx.x][threadIdx.y];
    }}
}}

void launch_naive(torch::Tensor input, torch::Tensor output, int M, int N) {{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    transpose_naive<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );
}}

void launch_tiled(torch::Tensor input, torch::Tensor output, int M, int N) {{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    transpose_tiled<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );
}}
"""

cpp_src = """
void launch_naive(torch::Tensor input, torch::Tensor output, int M, int N);
void launch_tiled(torch::Tensor input, torch::Tensor output, int M, int N);
"""

module = load_inline(
    name="transpose_compare",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["launch_naive", "launch_tiled"],
    verbose=False,
)

def transpose_naive(x):
    M, N = x.shape
    y = torch.zeros(N, M, device="cuda")
    module.launch_naive(x, y, M, N)
    return y

def transpose_tiled(x):
    M, N = x.shape
    y = torch.zeros(N, M, device="cuda")
    module.launch_tiled(x, y, M, N)
    return y

def benchmark(fn, x, iters=50):
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    start.record()
    for _ in range(iters):
        fn(x)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

if __name__ == "__main__":
    M = N = 4096
    x = torch.randn(M, N, device="cuda")

    ref = x.t()
    y_naive = transpose_naive(x)
    y_tiled = transpose_tiled(x)

    print("error naive:", (ref - y_naive).abs().max().item())
    print("error tiled:", (ref - y_tiled).abs().max().item())

    t_naive = benchmark(transpose_naive, x)
    t_tiled = benchmark(transpose_tiled, x)
    t_torch = benchmark(lambda v: v.t(), x)

    print()
    print("naive  ms:", round(t_naive, 3))
    print("tiled  ms:", round(t_tiled, 3))
    print("torch  ms:", round(t_torch, 3))
    print("speedup (tiled / naive):", round(t_naive / t_tiled, 2))
