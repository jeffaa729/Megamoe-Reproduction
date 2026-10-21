# MegaMoE on Hopper

BF16 MegaMoE-style expert-parallel forward kernels implemented with CUDA and CUTLASS/CuTe for NVLink-connected H100/H200 GPUs. Benchmarked against DeepEP V2 dispatch/combine + DeepGEMM BF16 grouped FC1/FC2 + weighted SwiGLU.

## Benchmark setup

V4-Flash-shaped routed MoE: `H=4096`, `I=2048`, `E=256`, `top-k=6`, SwiGLU clamp `10`. BF16 inputs, weights and outputs; FP32 accumulation. Fixed expert placement and precomputed routing.

H200 NVL, NVLink `NV6`, 600 W per GPU. Complete MoE forward measured with CUDA Graphs: median of 31 trials, 10 calls per replay, slowest Rank per sample. Reference % = baseline latency / custom latency × 100.

## 4× H200 NVL — EP4

64 experts/GPU. Measured on 2026-10-19 with PyTorch `2.13.0+cu130`, DeepEP `2.0.0+local` and DeepGEMM `2.8.2`.

| Tokens/GPU | Custom µs | DeepEP V2 + DeepGEMM µs | Reference % |
| ---: | ---: | ---: | ---: |
| 1 | 194.22 | 290.31 | 149.5 |
| 64 | 887.05 | 1068.29 | 120.4 |
| 256 | 910.07 | 1146.66 | 126.0 |
| 1024 | 1053.08 | 1440.81 | 136.8 |
| 2048 | 1852.50 | 2257.62 | 121.9 |
| 4096 | 3216.85 | 4013.21 | 124.8 |

## 2× H200 NVL — EP2

128 experts/GPU. Measured on 2026-10-18 with PyTorch `2.13.0+cu130`, DeepEP `2.0.0+af9a040` and DeepGEMM `2.8.1`.

| Tokens/GPU | Custom µs | DeepEP V2 + DeepGEMM µs | Reference % |
| ---: | ---: | ---: | ---: |
| 1 | 179.15 | 269.82 | 150.6 |
| 64 | 1662.81 | 2068.48 | 124.4 |
| 256 | 1739.71 | 2166.15 | 124.5 |
| 1024 | 1926.32 | 2481.00 | 128.8 |
| 2048 | 2024.23 | 2935.44 | 145.0 |
| 4096 | 3551.77 | 4588.19 | 129.2 |
