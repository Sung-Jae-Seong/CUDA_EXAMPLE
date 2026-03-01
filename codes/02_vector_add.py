import torch
from torch.utils.cpp_extension import load_inline

cuda_source = \
r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vector_add_launcher(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int n = a.numel();
    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );
}
'''

cpp_header = \
r'''
void vector_add_launcher(torch::Tensor a, torch::Tensor b, torch::Tensor c);
'''

module = load_inline(
    name='vector_add_ext',
    cpp_sources=cpp_header,
    cuda_sources=cuda_source,
    functions=['vector_add_launcher'],
    verbose=False,
)

def vector_add(a, b):
    c = torch.zeros_like(a)
    module.vector_add_launcher(a, b, c)
    return c

if __name__ == "__main__":
    device = 'cuda'
    N = 1_000_000
    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)

    c = vector_add(a, b)
    torch.cuda.synchronize()

    print(torch.allclose(c, a + b))
