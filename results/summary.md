Model `Qwen/Qwen2.5-0.5B-Instruct` on **Tesla T4**. Prompt 256 tokens, generate 128 tokens, batch 1, greedy, 10 timed runs per variant.

### Key findings

- **torch.compile + static KV cache: 4.2x decode throughput** (28.4 → 120.1 tok/s). Per-token latency fell from 34.1 ms to 8.2 ms, and p99 from 52 ms to 9.4 ms.
- **TensorRT engine: 2.8x faster prefill than PyTorch** (37.8 → 13.6 ms for a 256-token prompt, 1.16x vs torch.compile), with matching outputs (top-1 token match: yes, cosine similarity 0.9999).
- **Custom Triton fused add+RMSNorm kernel: 5.2x faster than eager PyTorch** (median over prefill-sized shapes), 1.02x vs torch.compile, reaching 72% of the GPU's peak memory bandwidth.
- **4-bit NF4 cut peak GPU memory 51%** (969 → 479 MB) at a 10% perplexity cost (15.98 → 17.63).
- **INT8 kept accuracy** (perplexity 16.01 vs 15.98) **but ran 3.9x slower** at batch 1: bitsandbytes LLM.int8 dequantizes weights on every matmul, so at batch 1 that overhead dominates.

### End-to-end generation

| Variant | Decode tok/s | Speedup | TTFT p50 (ms) | Per-token p50 (ms) | Per-token p99 (ms) | Peak memory (MB) | WikiText-2 perplexity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `eager_fp16` | 28.4 | 1.00x | 39.7 | 34.09 | 52.16 | 966 | 15.980 |
| `sdpa_fp16` | 23.8 | 0.84x | 42.7 | 34.92 | 159.10 | 969 | 15.980 |
| `triton_rmsnorm` | 29.2 | 1.03x | 40.5 | 33.18 | 48.53 | 969 | 15.980 |
| `compile` | 120.1 | 4.23x | 46.7 | 8.16 | 9.43 | 970 | 15.980 |
| `int8` | 7.3 | 0.26x | 164.8 | 135.01 | 177.76 | 628 | 16.012 |
| `nf4` | 21.1 | 0.74x | 62.3 | 45.95 | 66.51 | 479 | 17.629 |

![Benchmark summary](results/summary.png)

### TensorRT prefill (256-token prompt, TensorRT 11.4.0.106)

| Backend | p50 (ms) | p99 (ms) | Speedup |
|---|---:|---:|---:|
| PyTorch eager (SDPA) | 37.78 | 60.08 | 1.00x |
| torch.compile | 15.70 | 29.38 | 2.41x |
| TensorRT engine | 13.56 | 14.08 | 2.79x |

Output check vs PyTorch: top-1 token match **yes**, top-5 overlap 5/5, cosine similarity 0.99988. Engine build: 21 s, 4682 network layers.

### Triton kernel microbenchmark (Tesla T4, peak 320 GB/s)

M = rows (1 = one decode token, 4096 = long prefill), N = hidden size.

| M | N | Eager PyTorch (µs) | torch.compile (µs) | Triton (µs) | vs eager | vs compile | GB/s | % of peak |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 896 | 76.4 | 8.9 | 5.2 | 14.60x | 1.70x | 2 | 1% |
| 32 | 896 | 54.5 | 5.7 | 5.3 | 10.18x | 1.06x | 43 | 14% |
| 512 | 896 | 57.5 | 19.9 | 19.5 | 2.95x | 1.02x | 188 | 59% |
| 4096 | 896 | 684.2 | 128.0 | 153.1 | 4.47x | 0.84x | 192 | 60% |
| 1 | 2048 | 49.0 | 6.8 | 5.1 | 9.59x | 1.34x | 4 | 1% |
| 32 | 2048 | 85.0 | 8.6 | 7.6 | 11.16x | 1.13x | 69 | 22% |
| 512 | 2048 | 198.1 | 40.9 | 49.3 | 4.02x | 0.83x | 170 | 53% |
| 4096 | 2048 | 1524.5 | 297.3 | 292.8 | 5.21x | 1.02x | 229 | 72% |
| 1 | 4096 | 27.6 | 9.3 | 19.7 | 1.40x | 0.47x | 2 | 1% |
| 32 | 4096 | 38.4 | 10.9 | 8.4 | 4.60x | 1.30x | 126 | 40% |
| 512 | 4096 | 391.8 | 80.8 | 75.5 | 5.19x | 1.07x | 222 | 69% |
| 4096 | 4096 | 3021.6 | 599.6 | 585.2 | 5.16x | 1.02x | 229 | 72% |
