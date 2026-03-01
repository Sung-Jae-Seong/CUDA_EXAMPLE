# CUDA_EXAMPLE

This repo is written for people who mainly use PyTorch but want to understand “what’s really happening” on the GPU.

---

* 00_hello_world.py  
CUDA-kernel printf output of per-block and per-thread IDs; most basic JIT compilation example.

* 01_profiling.py  
Kernel runtime measurement comparison: incorrect method (time.time) vs correct method (CUDA events); asynchronous-execution pitfall demonstration.

* 02_vector_add.py  
Basic vector addition implementation: parallel addition of two vectors in a CUDA kernel.

* 03_RGB2GRAY.py  
RGB-to-grayscale conversion: HWC-format RGB image processing in a CUDA kernel.

* 04_cuda_indexing.py  
Indexing helper function example: grid-size calculation via integer-division-based cdiv.

* 05_device_query.py  
GPU device information and memory-usage printout; simple CUDA operation check.

* 06_grid_stride.py  
Vector addition comparison: monolithic approach vs grid-stride loop approach; scalability and performance difference demonstration.

* 07_coalescing.py  
Performance comparison experiment: coalesced vs strided vs random memory access patterns.

* 08_tiling.py  
Matrix multiplication performance comparison: shared-memory tiling vs naive implementation.

* 09_transposed_matrix.py  
Matrix transpose comparison: naive transpose vs shared-memory tile-based transpose; coalescing and bank-conflict effects demonstration.

* 10_vector_add.cu  
Vector addition kernel example written as a C++/CUDA (.cu) file.

* 11_nsys_profiling.py  
nsys timeline profiling using NVTX ranges.

* 12_cnn_optimize.py  
2D convolution benchmarking code: naive vs tiled vs constant-memory optimization.

* 13_cuda_reduction.py  
Comprehensive experiment code for parallel reduction optimizations (reduce0–reduce7); performance comparison.
