"""
naive 방식은 가장 직관적인 구현입니다. 각 스레드가 출력 픽셀 하나를 담당하고, 필요한 입력과 커널 값을 매번 글로벌 메모리에서 직접 읽습니다. 입력 채널 수 × 커널 크기²만큼 반복해서 글로벌 메모리를 접근하므로 메모리 트래픽이 매우 큽니다. 구현은 단순하지만 캐시 효율이 낮고, 대부분의 GPU에서 메모리 대역폭 병목으로 가장 느립니다.

tiling 방식은 입력 feature map을 shared memory에 타일 단위로 로드합니다. 한 블록이 TILE_SIZE×TILE_SIZE 영역과 halo를 포함한 입력을 shared memory에 올린 뒤, 같은 데이터를 여러 스레드가 재사용합니다. 그 결과 글로벌 메모리 읽기 횟수가 크게 줄고, arithmetic intensity가 증가합니다. 커널은 여전히 글로벌 메모리에서 읽지만, 입력 쪽 병목은 상당히 완화됩니다. 일반적으로 naive 대비 수 배 빠르며, 채널 수와 공간 해상도가 어느 정도 큰 경우 효과가 뚜렷합니다.

optimized 방식은 tiling에 더해 커널 가중치를 constant memory에 올립니다. constant memory는 동일 주소를 여러 스레드가 동시에 읽을 때 broadcast가 발생하므로, 작은 커널(여기서는 3×3)과 제한된 out_channels에서 매우 효율적입니다. 입력은 shared memory, 커널은 constant memory를 사용하므로 글로벌 메모리 접근이 최소화됩니다. 이 조건이 맞으면 세 구현 중 가장 빠릅니다. 다만 constant memory 크기 제약 때문에 in_channels ≤ 3, out_channels ≤ 64, kernel_size = 3 같은 조건이 필요합니다.
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

TILE_SIZE = 16
KERNEL_SIZE = 3

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_naive_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int batch, int in_channels, int out_channels,
    int height, int width,
    int kernel_size
) {
    int oc = blockIdx.z;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;

    if (oc >= out_channels || oy >= height || ox >= width) return;

    int pad = kernel_size / 2;
    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ic++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int iy = oy + ky - pad;
                int ix = ox + kx - pad;

                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    int input_idx = ic * height * width + iy * width + ix;
                    int kernel_idx = oc * in_channels * kernel_size * kernel_size +
                                     ic * kernel_size * kernel_size +
                                     ky * kernel_size + kx;
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
    }

    int output_idx = oc * height * width + oy * width + ox;
    output[output_idx] = sum;
}

__global__ void conv2d_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int batch, int in_channels, int out_channels,
    int height, int width,
    int kernel_size
) {
    const int TILE_SIZE = 16;
    const int HALO = 1;
    const int SHARED_SIZE = TILE_SIZE + 2 * HALO;
    __shared__ float tile[SHARED_SIZE][SHARED_SIZE];

    int oc = blockIdx.z;
    int tile_y = blockIdx.y * TILE_SIZE;
    int tile_x = blockIdx.x * TILE_SIZE;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int oy = tile_y + ty;
    int ox = tile_x + tx;

    if (oc >= out_channels) return;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ic++) {
        for (int i = ty; i < SHARED_SIZE; i += blockDim.y) {
            for (int j = tx; j < SHARED_SIZE; j += blockDim.x) {
                int iy = tile_y + i - HALO;
                int ix = tile_x + j - HALO;

                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    int input_idx = ic * height * width + iy * width + ix;
                    tile[i][j] = input[input_idx];
                } else {
                    tile[i][j] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (oy < height && ox < width) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int kernel_idx = oc * in_channels * kernel_size * kernel_size +
                                     ic * kernel_size * kernel_size +
                                     ky * kernel_size + kx;
                    sum += tile[ty + ky][tx + kx] * kernel[kernel_idx];
                }
            }
        }
        __syncthreads();
    }

    if (oy < height && ox < width) {
        int output_idx = oc * height * width + oy * width + ox;
        output[output_idx] = sum;
    }
}

__constant__ float const_kernel[64 * 3 * 3 * 3];

__global__ void conv2d_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch, int in_channels, int out_channels,
    int height, int width,
    int kernel_size
) {
    const int TILE_SIZE = 16;
    const int HALO = 1;
    const int SHARED_SIZE = TILE_SIZE + 2 * HALO;
    __shared__ float tile[SHARED_SIZE][SHARED_SIZE];

    int oc = blockIdx.z;
    int tile_y = blockIdx.y * TILE_SIZE;
    int tile_x = blockIdx.x * TILE_SIZE;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int oy = tile_y + ty;
    int ox = tile_x + tx;

    if (oc >= out_channels) return;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ic++) {
        for (int i = ty; i < SHARED_SIZE; i += blockDim.y) {
            for (int j = tx; j < SHARED_SIZE; j += blockDim.x) {
                int iy = tile_y + i - HALO;
                int ix = tile_x + j - HALO;

                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    tile[i][j] = input[ic * height * width + iy * width + ix];
                } else {
                    tile[i][j] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (oy < height && ox < width) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int kernel_idx = oc * in_channels * kernel_size * kernel_size +
                                     ic * kernel_size * kernel_size +
                                     ky * kernel_size + kx;
                    sum += tile[ty + ky][tx + kx] * const_kernel[kernel_idx];
                }
            }
        }
        __syncthreads();
    }

    if (oy < height && ox < width) {
        output[oc * height * width + oy * width + ox] = sum;
    }
}

void launch_conv2d_naive(
    torch::Tensor input, torch::Tensor kernel, torch::Tensor output,
    int batch, int in_channels, int out_channels, int height, int width, int kernel_size
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16, out_channels);
    conv2d_naive_kernel<<<grid, block>>>(
        input.data_ptr<float>(), kernel.data_ptr<float>(), output.data_ptr<float>(),
        batch, in_channels, out_channels, height, width, kernel_size
    );
}

void launch_conv2d_tiled(
    torch::Tensor input, torch::Tensor kernel, torch::Tensor output,
    int batch, int in_channels, int out_channels, int height, int width, int kernel_size
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16, out_channels);
    conv2d_tiled_kernel<<<grid, block>>>(
        input.data_ptr<float>(), kernel.data_ptr<float>(), output.data_ptr<float>(),
        batch, in_channels, out_channels, height, width, kernel_size
    );
}

void launch_conv2d_optimized(
    torch::Tensor input, torch::Tensor kernel, torch::Tensor output,
    int batch, int in_channels, int out_channels, int height, int width, int kernel_size
) {
    cudaMemcpyToSymbol(const_kernel, kernel.data_ptr<float>(),
                       kernel.numel() * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16, out_channels);
    conv2d_optimized_kernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, in_channels, out_channels, height, width, kernel_size
    );
}
"""

CPP_HDR = r"""
void launch_conv2d_naive(
    torch::Tensor input, torch::Tensor kernel, torch::Tensor output,
    int batch, int in_channels, int out_channels, int height, int width, int kernel_size
);
void launch_conv2d_tiled(
    torch::Tensor input, torch::Tensor kernel, torch::Tensor output,
    int batch, int in_channels, int out_channels, int height, int width, int kernel_size
);
void launch_conv2d_optimized(
    torch::Tensor input, torch::Tensor kernel, torch::Tensor output,
    int batch, int in_channels, int out_channels, int height, int width, int kernel_size
);
"""

def build_extension():
    return load_inline(
        name="conv2d_module",
        cpp_sources=[CPP_HDR],
        cuda_sources=[CUDA_SRC],
        functions=["launch_conv2d_naive", "launch_conv2d_tiled", "launch_conv2d_optimized"],
        verbose=False,
    )

def conv2d_level1(module, x3d, w4d):
    c, h, w = x3d.shape
    oc, _, kh, kw = w4d.shape
    y = torch.empty((oc, h, w), device="cuda", dtype=torch.float32)
    module.launch_conv2d_naive(x3d.contiguous(), w4d.contiguous(), y, 1, c, oc, h, w, kh)
    return y

def conv2d_level2(module, x3d, w4d):
    c, h, w = x3d.shape
    oc, _, kh, kw = w4d.shape
    y = torch.empty((oc, h, w), device="cuda", dtype=torch.float32)
    module.launch_conv2d_tiled(x3d.contiguous(), w4d.contiguous(), y, 1, c, oc, h, w, kh)
    return y

def conv2d_level3(module, x3d, w4d):
    c, h, w = x3d.shape
    oc, _, kh, kw = w4d.shape
    y = torch.empty((oc, h, w), device="cuda", dtype=torch.float32)
    module.launch_conv2d_optimized(x3d.contiguous(), w4d.contiguous(), y, 1, c, oc, h, w, kh)
    return y

def make_inputs(input_shape, out_channels, kernel_size):
    c, h, w = input_shape
    x = torch.randn((1, c, h, w), device="cuda", dtype=torch.float32)
    wgt = torch.randn((out_channels, c, kernel_size, kernel_size), device="cuda", dtype=torch.float32)
    return x, wgt

def reference_conv2d(x4d, w4d, pad):
    return F.conv2d(x4d, w4d, padding=pad)

def check_correctness(module, input_shape=(3, 224, 224), out_channels=64, kernel_size=3):
    x4d, w4d = make_inputs(input_shape, out_channels, kernel_size)
    pad = kernel_size // 2
    ref = reference_conv2d(x4d, w4d, pad).squeeze(0)

    x3d = x4d.squeeze(0)

    y1 = conv2d_level1(module, x3d, w4d)
    y2 = conv2d_level2(module, x3d, w4d)
    y3 = conv2d_level3(module, x3d, w4d)

    def stat(y):
        ok = torch.allclose(y, ref, rtol=1e-3, atol=1e-3)
        err = (y - ref).abs().max().item()
        return ok, err

    ok1, e1 = stat(y1)
    ok2, e2 = stat(y2)
    ok3, e3 = stat(y3)

    print("correctness")
    print(f"  level1  ok={ok1}  max_err={e1:.2e}")
    print(f"  level2  ok={ok2}  max_err={e2:.2e}")
    print(f"  level3  ok={ok3}  max_err={e3:.2e}")
    print()

def time_ms(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def benchmark(module, input_shape, out_channels, kernel_size, iters):
    x4d, w4d = make_inputs(input_shape, out_channels, kernel_size)
    pad = kernel_size // 2
    x3d = x4d.squeeze(0)

    def f0():
        return reference_conv2d(x4d, w4d, pad)

    def f1():
        return conv2d_level1(module, x3d, w4d)

    def f2():
        return conv2d_level2(module, x3d, w4d)

    def f3():
        return conv2d_level3(module, x3d, w4d)

    # Level 3 사용 가능 조건
    enable_level3 = (
        x3d.shape[0] <= 3 and
        out_channels <= 64 and
        kernel_size == 3
    )

    # warm-up
    for _ in range(10):
        f0(); f1(); f2()
        if enable_level3:
            f3()
    torch.cuda.synchronize()

    # timing
    t0 = time_ms(f0, iters)
    t1 = time_ms(f1, iters)
    t2 = time_ms(f2, iters)
    t3 = time_ms(f3, iters) if enable_level3 else None

    def pct(t):
        return (t0 / t) * 100.0

    print(f"benchmark  input={input_shape}  out_ch={out_channels}  k={kernel_size}  iters={iters}")
    print(f"  pytorch  {t0:10.4f} ms   100.0%")
    print(f"  level1   {t1:10.4f} ms   {pct(t1):6.1f}%")
    print(f"  level2   {t2:10.4f} ms   {pct(t2):6.1f}%")

    results = [("level1", t1), ("level2", t2)]

    if enable_level3:
        print(f"  level3   {t3:10.4f} ms   {pct(t3):6.1f}%")
        results.append(("level3", t3))
    else:
        print("  level3   skipped (constant memory constraint)")

    best = min(results, key=lambda x: x[1])
    print(f"  best     {best[0]}  {pct(best[1]):.1f}% of pytorch")
    print()


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(torch.cuda.get_device_name(0))
    module = build_extension()

    check_correctness(module, input_shape=(3, 224, 224), out_channels=64, kernel_size=3)

    benchmark(module, input_shape=(3, 32, 32), out_channels=16, kernel_size=3, iters=100)
    # 거의 비슷 -> 연산 크기가 작기 때문에 launch비용이 차지하는 부분이 크고 pytorch와 거의 비슷하다.
    benchmark(module, input_shape=(64, 56, 56), out_channels=128, kernel_size=3, iters=50)
    # input channel이 큰 경우 global memory에 자주 접근하여 느리다.
    # level2도 input channel마다 global memory에 접근하여 느리다.
    # level3는 constant memory size 제약 때문에 어렵다.
    benchmark(module, input_shape=(3, 1024, 1024), out_channels=64, kernel_size=3, iters=50)
    # input channel이 작고 H*W가 큰 경우 constant memory의 효율이 좋아진다.

if __name__ == "__main__":
    main()
