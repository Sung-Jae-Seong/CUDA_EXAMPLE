import math
import torch
from torch.utils.cpp_extension import load_inline

def cdiv(a, b):
    return (a + b - 1) // b

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFF, v, offset);
    }
    return v;
}

template <unsigned int blockSize>
__device__ __forceinline__ void warpReduce(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64)  sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32)  sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16)  sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)   sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)   sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)   sdata[tid] += sdata[tid + 1];
}

__global__ void reduce0(const float* __restrict__ g_in, float* __restrict__ g_out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < (unsigned)n) ? g_in[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
        if ((tid % (2 * s)) == 0) {
            if (tid + s < blockDim.x) sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

__global__ void reduce1(const float* __restrict__ g_in, float* __restrict__ g_out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < (unsigned)n) ? g_in[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
        unsigned int index = 2 * s * tid;
        if (index + s < blockDim.x) sdata[index] += sdata[index + s];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

__global__ void reduce2(const float* __restrict__ g_in, float* __restrict__ g_out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < (unsigned)n) ? g_in[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

__global__ void reduce3(const float* __restrict__ g_in, float* __restrict__ g_out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int base = blockIdx.x * (blockDim.x * 2) + tid;

    float a = (base < (unsigned)n) ? g_in[base] : 0.0f;
    float b = (base + blockDim.x < (unsigned)n) ? g_in[base + blockDim.x] : 0.0f;
    sdata[tid] = a + b;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

__global__ void reduce4(const float* __restrict__ g_in, float* __restrict__ g_out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int base = blockIdx.x * (blockDim.x * 2) + tid;

    float a = (base < (unsigned)n) ? g_in[base] : 0.0f;
    float b = (base + blockDim.x < (unsigned)n) ? g_in[base + blockDim.x] : 0.0f;
    sdata[tid] = a + b;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce5(const float* __restrict__ g_in, float* __restrict__ g_out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int base = blockIdx.x * (blockSize * 2) + tid;

    float a = (base < (unsigned)n) ? g_in[base] : 0.0f;
    float b = (base + blockSize < (unsigned)n) ? g_in[base + blockSize] : 0.0f;
    sdata[tid] = a + b;
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) sdata[tid] += sdata[tid + 512]; __syncthreads(); }
    if (blockSize >= 512)  { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256)  { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128)  { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }

    if (tid < 32) {
        warpReduce<blockSize>((volatile float*)sdata, tid);
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce6(const float* __restrict__ g_in, float* __restrict__ g_out, int n) {
    float sum = 0.0f;
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    while (idx < (unsigned)n) {
        sum += g_in[idx];
        if (idx + blockSize < (unsigned)n) sum += g_in[idx + blockSize];
        idx += gridSize;
    }

    extern __shared__ float sdata[];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockSize >> 1; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

__global__ void reduce7_warp_shuffle(const float* __restrict__ g_in, float* __restrict__ g_out, int n) {
    float sum = 0.0f;
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    while (idx < (unsigned)n) {
        sum += g_in[idx];
        if (idx + blockDim.x < (unsigned)n) sum += g_in[idx + blockDim.x];
        idx += gridSize;
    }

    sum = warp_reduce_sum(sum);

    __shared__ float warp_sums[32];
    int lane = tid & 31;
    int warp = tid >> 5;

    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (warp == 0) {
        block_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) g_out[blockIdx.x] = block_sum;
    }
}

void launch_reduce0(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads) {
    reduce0<<<blocks, threads, threads * sizeof(float)>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
}
void launch_reduce1(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads) {
    reduce1<<<blocks, threads, threads * sizeof(float)>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
}
void launch_reduce2(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads) {
    reduce2<<<blocks, threads, threads * sizeof(float)>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
}
void launch_reduce3(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads) {
    reduce3<<<blocks, threads, threads * sizeof(float)>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
}
void launch_reduce4(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads) {
    reduce4<<<blocks, threads, threads * sizeof(float)>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
}

void launch_reduce5(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads) {
    size_t sh = threads * sizeof(float);
    if (threads == 1024) reduce5<1024><<<blocks, 1024, sh>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    else if (threads == 512) reduce5<512><<<blocks, 512, sh>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    else if (threads == 256) reduce5<256><<<blocks, 256, sh>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    else if (threads == 128) reduce5<128><<<blocks, 128, sh>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    else reduce5<256><<<blocks, 256, 256 * sizeof(float)>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
}

void launch_reduce6(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads) {
    size_t sh = threads * sizeof(float);
    if (threads == 1024) reduce6<1024><<<blocks, 1024, sh>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    else if (threads == 512) reduce6<512><<<blocks, 512, sh>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    else if (threads == 256) reduce6<256><<<blocks, 256, sh>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    else if (threads == 128) reduce6<128><<<blocks, 128, sh>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    else reduce6<256><<<blocks, 256, 256 * sizeof(float)>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
}

void launch_reduce7(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads) {
    reduce7_warp_shuffle<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
}
"""

CPP_HDR = r"""
void launch_reduce0(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads);
void launch_reduce1(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads);
void launch_reduce2(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads);
void launch_reduce3(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads);
void launch_reduce4(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads);
void launch_reduce5(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads);
void launch_reduce6(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads);
void launch_reduce7(torch::Tensor x, torch::Tensor y, int n, int blocks, int threads);
"""

module = load_inline(
    name="reduction_suite_ext",
    cpp_sources=[CPP_HDR],
    cuda_sources=[CUDA_SRC],
    functions=[
        "launch_reduce0",
        "launch_reduce1",
        "launch_reduce2",
        "launch_reduce3",
        "launch_reduce4",
        "launch_reduce5",
        "launch_reduce6",
        "launch_reduce7",
    ],
    extra_cuda_cflags=["--use_fast_math"],
    verbose=False,
)

def make_plan(n, threads, first_stage_two_load):
    sizes = [n]
    if first_stage_two_load:
        sizes.append(cdiv(sizes[-1], threads * 2))
    else:
        sizes.append(cdiv(sizes[-1], threads))
    while sizes[-1] > 1:
        sizes.append(cdiv(sizes[-1], threads * 2))
    return sizes

def alloc_buffers(sizes, device):
    bufs = []
    for s in sizes[1:]:
        bufs.append(torch.empty((s,), device=device, dtype=torch.float32))
    return bufs

def run_full_reduction(kind, x, bufs, threads, sm_blocks_for_reduce6=None):
    n0 = x.numel()

    if kind in ("reduce0", "reduce1", "reduce2"):
        blocks0 = bufs[0].numel()
        getattr(module, f"launch_{kind}")(x, bufs[0], n0, blocks0, threads)
    elif kind in ("reduce3", "reduce4", "reduce5"):
        blocks0 = bufs[0].numel()
        getattr(module, f"launch_{kind}")(x, bufs[0], n0, blocks0, threads)
    elif kind == "reduce6":
        blocks0 = bufs[0].numel()
        blocks = sm_blocks_for_reduce6 if sm_blocks_for_reduce6 is not None else blocks0
        blocks = min(blocks, blocks0)
        module.launch_reduce6(x, bufs[0], n0, blocks, threads)
    elif kind == "reduce7":
        blocks0 = bufs[0].numel()
        module.launch_reduce7(x, bufs[0], n0, blocks0, threads)
    else:
        raise ValueError(kind)

    cur = bufs[0]
    for i in range(1, len(bufs)):
        out = bufs[i]
        blocks = out.numel()
        module.launch_reduce7(cur, out, cur.numel(), blocks, threads)
        cur = out
    return cur

def time_ms(fn, iters):
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

def gbps(n, ms):
    bytes_moved = n * 4
    return (bytes_moved / 1e9) / (ms / 1e3)

def main():
    assert torch.cuda.is_available()
    device = "cuda"
    props = torch.cuda.get_device_properties(0)
    sm = props.multi_processor_count

    N = 1 << 26
    threads = 256
    iters = 50

    x = torch.randn((N,), device=device, dtype=torch.float32)

    ref = x.sum()
    torch.cuda.synchronize()

    kinds = [
        ("naive_interleaved_divergent", "reduce0", False),
        ("interleaved_bank_conflict", "reduce1", False),
        ("sequential_addressing", "reduce2", False),
        ("first_add_during_load", "reduce3", True),
        ("unroll_last_warp", "reduce4", True),
        ("completely_unroll", "reduce5", True),
        ("multiple_adds_threads", "reduce6", True),
        ("warp_shuffle", "reduce7", True),
        ("pytorch_baseline", "torch", None),
    ]

    results = []

    for name, kind, two_load in kinds:
        if kind == "torch":
            for _ in range(10):
                _ = x.sum()
            torch.cuda.synchronize()

            ms = time_ms(lambda: x.sum(), iters)
            out = x.sum()
            ok = torch.allclose(out, ref, rtol=1e-4, atol=1e-4)
            results.append((name, ms, gbps(N, ms), ok))
            continue

        plan_sizes = make_plan(N, threads, two_load)
        bufs = alloc_buffers(plan_sizes, device)

        if kind == "reduce6":
            blocks0 = bufs[0].numel()
            sm_blocks = sm * 4
            sm_blocks = max(1, min(sm_blocks, blocks0))
            warm = lambda: run_full_reduction(kind, x, bufs, threads, sm_blocks_for_reduce6=sm_blocks)
            for _ in range(10):
                warm()
            torch.cuda.synchronize()
            ms = time_ms(warm, iters)
            out = warm()[0]
        else:
            warm = lambda: run_full_reduction(kind, x, bufs, threads)
            for _ in range(10):
                warm()
            torch.cuda.synchronize()
            ms = time_ms(warm, iters)
            out = warm()[0]

        ok = torch.allclose(out, ref, rtol=1e-3, atol=1e-2)
        results.append((name, ms, gbps(N, ms), ok))

    print("")
    print(f"GPU: {props.name}")
    print(f"N={N} threads={threads} iters={iters}")
    print("")
    print(f"{'name':<30} {'ms/iter':>10} {'GB/s':>10} {'correct':>8}")

    for name, ms, g, ok in results:
        print(f"{name:<30} {ms:10.4f} {g:10.2f} {str(ok):>8}")

if __name__ == "__main__":
    main()
