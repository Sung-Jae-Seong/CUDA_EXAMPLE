import torch
from torch.utils.cpp_extension import load_inline


def cdiv(n: int, d: int) -> int:
    return (n + d - 1) // d


cuda_src = r"""
#include <torch/extension.h>

__global__ void vec_add_mono(
    const float* a, const float* b, float* c, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vec_add_stride(
    const float* a, const float* b, float* c, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        c[i] = a[i] + b[i];
}

void launch_mono(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                 int n, int blocks, int threads) {
    vec_add_mono<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n
    );
}

void launch_stride(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                   int n, int blocks, int threads) {
    vec_add_stride<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n
    );
}
"""

cpp_hdr = r"""
void launch_mono(torch::Tensor, torch::Tensor, torch::Tensor,
                 int, int, int);
void launch_stride(torch::Tensor, torch::Tensor, torch::Tensor,
                   int, int, int);
"""

module = load_inline(
    name="vec_add",
    cpp_sources=[cpp_hdr],
    cuda_sources=[cuda_src],
    functions=["launch_mono", "launch_stride"],
    verbose=False,
)

def run_mono(a, b, threads=256):
    n = a.numel()
    c = torch.empty_like(a)
    blocks = cdiv(n, threads)
    module.launch_mono(a, b, c, n, blocks, threads)
    return c

def run_stride(a, b, threads=256, blocks=None):
    n = a.numel()
    c = torch.empty_like(a)

    if blocks is None:
        sm = torch.cuda.get_device_properties(0).multi_processor_count
        blocks = sm * 4

    module.launch_stride(a, b, c, n, blocks, threads)
    return c

def benchmark(n=10_000_000, iters=50):
    a = torch.rand(n, device="cuda")
    b = torch.rand(n, device="cuda")

    ref = a + b

    for fn in (run_mono, run_stride):
        fn(a, b)
    torch.cuda.synchronize()

    def time(fn):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            c = fn(a, b)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters, torch.allclose(c, ref)

    t_mono, ok_mono = time(run_mono)
    t_stride, ok_stride = time(run_stride)

    print(f"Monolithic : {t_mono:.3f} ms | correct = {ok_mono}")
    print(f"GridStride : {t_stride:.3f} ms | correct = {ok_stride}")
    print(f"Speedup    : {t_mono / t_stride:.2f}x")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    benchmark()
