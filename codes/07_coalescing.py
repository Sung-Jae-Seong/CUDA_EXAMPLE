import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r'''
#include <torch/extension.h>

__global__ void coalesced(
    const float* x, float* y, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] * 2.0f;
    }
}

__global__ void strided(
    const float* x, float* y, int n, int stride
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int idx = (i * stride) % n;
        y[i] = x[idx] * 2.0f;
    }
}

__global__ void random_access(
    const float* x, float* y, const int* idx, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[idx[i]] * 2.0f;
    }
}

void launch_coalesced(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads) {
    coalesced<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
}

void launch_strided(torch::Tensor x, torch::Tensor y, int n, int stride, int blocks, int threads) {
    strided<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n, stride);
}

void launch_random(torch::Tensor x, torch::Tensor y, torch::Tensor idx, int n, int blocks, int threads) {
    random_access<<<blocks, threads>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), idx.data_ptr<int>(), n
    );
}
'''

cpp_hdr = r'''
void launch_coalesced(torch::Tensor, torch::Tensor, int, int, int);
void launch_strided(torch::Tensor, torch::Tensor, int, int, int, int);
void launch_random(torch::Tensor, torch::Tensor, torch::Tensor, int, int, int);
'''

module = load_inline(
    name="mem_access_demo",
    cpp_sources=[cpp_hdr],
    cuda_sources=[cuda_src],
    functions=["launch_coalesced", "launch_strided", "launch_random"],
    verbose=False,
)

def benchmark(fn, iters=50):
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters

N = 10_000_000
threads = 256
blocks = (N + threads - 1) // threads
stride = 4

x = torch.rand(N, device="cuda")
y = torch.empty_like(x)
idx = torch.randint(0, N, (N,), device="cuda", dtype=torch.int32)

t_coalesced = benchmark(
    lambda: module.launch_coalesced(x, y, N, blocks, threads)
)

t_strided = benchmark(
    lambda: module.launch_strided(x, y, N, stride, blocks, threads)
)

t_random = benchmark(
    lambda: module.launch_random(x, y, idx, N, blocks, threads)
)

print("Memory Access Pattern Performance\n")
print(f"Coalesced access : {t_coalesced:.3f} ms")
print(f"Strided access   : {t_strided:.3f} ms  ({t_strided / t_coalesced:.1f}x slower)")
print(f"Random access    : {t_random:.3f} ms  ({t_random / t_coalesced:.1f}x slower)")
