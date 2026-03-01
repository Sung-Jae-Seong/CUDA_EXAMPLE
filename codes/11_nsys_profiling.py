import torch
import time

torch.cuda.nvtx.range_push("Data Loading")
data = torch.randn(1024, 1024, device="cuda")
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("Compute")
for _ in range(10):
    data = torch.mm(data, data)
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("Sync")
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

time.sleep(1)
print("done")

# nsys profile --trace=cuda,nvtx -o basic_profile /usr/bin/python3 11_nsys_profiling.py

# which nsys-ui
# /usr/local/cuda-12.4/bin/nsys-ui
# nsys-ui basic_profile.nsys-rep
