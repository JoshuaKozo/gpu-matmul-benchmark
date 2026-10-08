# GPU Matrix Multiplication Benchmark

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoshuaKozo/gpu-matmul-benchmark/blob/main/gpu_matmul_benchmark.ipynb)

I built this project to learn how computation differs between CPUs and GPUs, and how those differences allow GPUs to accelerate AI. I compared matrix multiplication four ways on an NVIDIA Tesla T4: a pure Python triple loop, NumPy on the CPU, CuPy on the GPU, and a custom CUDA kernel I wrote with Numba.

## Key results

- CuPy (GPU) was **12,212×** faster than a pure Python triple loop at 256×256
- CuPy was up to **42×** faster than NumPy on the CPU (2048×2048), and **17.5×** faster at 4096×4096 even including CPU↔GPU data transfer
- The GPU only beat the CPU above **128×128** (compute only) or **512×512** (including data transfer)
- My custom CUDA kernel was **2.1×** faster than NumPy at 4096×4096, and making its memory access coalesced **doubled** its speed with no change to the math

![Runtime and speedup charts](benchmark_results.png)

## Why matrix multiplication?

Matrix multiplication is an extremely useful benchmark, as it is the basis of training and running the neural networks used in AI, and its O(n³) work makes it a demanding test of how fast hardware can compute. GPUs are uniquely suited for large-scale matrix multiplication as a result of each entry in the output matrix C being an independent dot product. This allows each entry in C to be computed by its own thread. A 4096×4096 multiply in this project launched about 16.7 million threads, spread across the T4's 2,560 cores.

## What I built

- **Pure Python triple loop:** Uses three nested for loops: the `i` and `j` loops pick each entry of the output matrix C, and the `k` loop computes that entry's dot product, resulting in O(n³) work (about 1.2 seconds at just 256×256).
- **NumPy (CPU):** Does the same O(n³) math, but much faster because NumPy uses BLAS (Basic Linear Algebra Subprograms), an optimized library that runs compiled code, works in cache-sized blocks, uses SIMD (single instruction, multiple data) instructions, and runs across multiple CPU cores, making it about 3,300× faster than the Python loop at 256×256.
- **CuPy (GPU):** A NumPy-compatible library that copies the matrices into GPU memory and runs the multiplication with NVIDIA's cuBLAS library, up to 42× faster than NumPy.
- **Custom CUDA kernel (Numba):** I wrote a kernel, compiled with Numba's `@cuda.jit`, that turns the `i` and `j` loops into a grid of GPU threads, where each thread corresponds to one entry of the output matrix C and runs only the `k` loop. Each thread does O(n) work, but with n² threads the total is still O(n³). The speedup comes from running thousands of threads in parallel. It was 2.1× faster than NumPy at 4096×4096.

All four are checked against NumPy with `np.allclose` before timing. Matrices are random `float32`, 8 sizes from 32×32 to 4096×4096.

## How I measured

[3–4 short bullets: warm-up run first; `synchronize()` before stopping the timer (and why); median of 5 runs; GPU methods timed with and without CPU↔GPU data transfer.]
- 
## Results

Median time in milliseconds:

| n | Python | NumPy | CuPy | CuPy + transfer | Kernel | Kernel + transfer |
|---|---|---|---|---|---|---|
| 32 | 1.883 | 0.004 | 0.037 | 0.176 | 0.051 | 0.663 |
| 64 | 15.489 | 0.008 | 0.038 | 0.178 | 0.051 | 0.605 |
| 128 | 129.870 | 0.065 | 0.055 | 0.235 | 0.107 | 0.866 |
| 256 | 1203.627 | 0.361 | 0.099 | 0.380 | 0.379 | 1.374 |
| 512 | — | 2.610 | 0.180 | 0.897 | 2.331 | 4.088 |
| 1024 | — | 17.033 | 0.883 | 3.181 | 17.843 | 18.686 |
| 2048 | — | 136.063 | 3.212 | 13.581 | 79.136 | 86.545 |
| 4096 | — | 1308.471 | 47.608 | 74.941 | 633.900 | 687.471 |

## What I learned

**[Title: The GPU isn't always faster]**
[2–3 sentences: fixed launch overhead at small sizes, and how the data transfer pushes the break-even point from 128 to 512.]

**[Title: Memory access matters as much as the math]**
[2–3 sentences: swapping `row, col` to `col, row` in `cuda.grid(2)` made the kernel 2× faster (1.27 s → 0.63 s at 4096). Explain warps and coalesced memory access in plain language.]

## Limitations and next steps

[3–4 bullets: e.g., one GPU model on a shared Colab machine; the T4 likely throttled during the longest runs; next, a shared-memory tiled kernel or writing the kernel in CUDA C++.]

## How to run

Click the **Open in Colab** badge above, select **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.

To run locally (requires an NVIDIA GPU):

```bash
pip install -r requirements.txt
jupyter notebook gpu_matmul_benchmark.ipynb
```
