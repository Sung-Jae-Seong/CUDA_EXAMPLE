import os
import time
import torch
from torch.utils.cpp_extension import load_inline

# 1. CUDA 소스 (구현부)
cuda_source = """
#include <torch/extension.h>
#include <cstdio>

__global__ void hello_kernel() {
    if(threadIdx.x == 0){
        printf("🚀 Hello from CUDA Kernel! Block %d Thread %d\\n", blockIdx.x, threadIdx.x);
    }
}

void hello(int blocks, int threads) {
    hello_kernel<<<blocks, threads>>>();
    cudaDeviceSynchronize();
}
"""

# 2. C++ 헤더 (선언부) - 중요!
# 컴파일러에게 "hello라는 함수가 있어"라고 미리 알려줍니다.
cpp_header = "void hello(int blocks, int threads);"

# JIT 컴파일
my_module = load_inline(
    name='hello_extension',
    cpp_sources=[cpp_header],    # 선언부
    cuda_sources=[cuda_source],  # 구현부
    functions=['hello'],         # 'hello_kernel'은 지워야 합니다!
    verbose=True
)

# 잘못된 측정 (Wrong)
start = time.time()
my_module.hello(100, 100)  # CPU는 명령만 내리고 바로 통과
print(f"Time: {time.time() - start} ms")  # 거의 0초가 나옴

# 올바른 측정 (Correct)
torch.cuda.synchronize()  # 기존 작업 대기
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
my_module.hello(100, 100)
end.record()

torch.cuda.synchronize()  # GPU 작업 완료 대기
print(f"Elapsed time: {start.elapsed_time(end)} ms")