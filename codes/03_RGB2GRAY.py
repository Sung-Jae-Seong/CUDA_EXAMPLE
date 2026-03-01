import torch
from torch.utils.cpp_extension import load_inline
import matplotlib.pyplot as plt

# global은 void만 가능

cuda_source = \
r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void rgb_to_gray_kernel(const float* input, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float r = input[i * 3 + 0];
        float g = input[i * 3 + 1];
        float b = input[i * 3 + 2];
        out[i] = 0.21f * r + 0.72f * g + 0.07f * b;
    }
}

torch::Tensor rgb_to_gray(torch::Tensor input) {
    auto H = (int)input.size(0);
    auto W = (int)input.size(1);
    int n = H * W;

    auto output = torch::empty({H, W}, input.options().dtype(torch::kFloat32));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    rgb_to_gray_kernel<<<blocks, threads>>>(
        (const float*)input.data_ptr<float>(),
        (float*)output.data_ptr<float>(),
        n
    );

    return output;
}
'''

cpp_header = \
r'''
torch::Tensor rgb_to_gray(torch::Tensor input);
'''

module = load_inline(
    name='rgb_to_gray_cpp',
    cpp_sources=cpp_header,
    cuda_sources=cuda_source,
    functions=['rgb_to_gray'],
    verbose=False,
)

def rgb_to_gray(img_hwc):
    return module.rgb_to_gray(img_hwc)

if __name__ == "__main__":
    device = "cuda"
    H, W = 1080, 1920
    img = torch.zeros((H, W, 3), device=device, dtype=torch.float32)
    w3 = W // 3
    img[:, :w3, 0] = 255.0
    img[:, w3:2*w3, 1] = 255.0
    img[:, 2*w3:, 2] = 255.0
    img = img.contiguous()
    gray = rgb_to_gray(img)
    torch.cuda.synchronize()
    print(img.shape, gray.shape, gray.dtype)

    img_cpu = img.byte().cpu().numpy()
    gray_cpu = gray.cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("RGB")
    plt.imshow(img_cpu)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Grayscale")
    plt.imshow(gray_cpu, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()