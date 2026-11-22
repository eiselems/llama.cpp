# Theoretical ceilings (2026-11-22)

Method from github.com/sybil-solutions/inference-optimisation-skill: measure the hardware, take the model's real
tensor sizes, compute the ceiling, report progress as a share of it.

## Measured hardware (RTX 3070 8 GB, Ryzen 5 5600X, dual-channel DDR4)

| what | measured | how |
|---|---:|---|
| GPU memory bandwidth | 404.6 GB/s | 1 GiB device-to-device copy (spec 448) |
| PCIe upload, pinned host memory | 26.3 GB/s | 1 GiB host-to-device copy (link: PCIe 4.0 x16) |
| CPU RAM read bandwidth | 48.6 GB/s | 4 GiB read, 6 threads |
| GPU FP16 GEMM, FP32 accumulate | 43-46 TFLOPS | cuBLAS 4096^3 / 8192^3 (FP16 accumulate: 85) |

Practical compute = 50 % of 44 TFLOPS = 22 TFLOPS.

## Model (Inkling-2-Mini UD-Q4_K_M, from the GGUF tensors)

48 layer applications per token (layers 6-17 run 3 times), 8 of 128 experts each.

| per token | size |
|---|---:|
| dense weights read (attention, norms, router, loop control) | 1.66 GiB |
| active experts (8/128 per application) | 1.66 GiB |
| LM head | 0.46 GiB |
| linear-layer FLOPs | 8.90 GFLOP (+ attention, 9.64 total at a 4.8k prompt) |

All experts: 13.76 GiB. One prefill ubatch with op offload uploads 26.58 GiB (every expert tensor of every
application, the looped ones three times).

## Decode ceilings (short context, one user)

| scenario | ceiling |
|---|---:|
| all experts in GPU memory (needs a ~16 GB card) | 99.6 t/s |
| all experts computed by the CPU from RAM, no cache | 23.6 t/s |
| our cache mix, every part serial (83 % hits; misses 70 % CPU / 30 % uploaded) | 58.4 t/s |
| our cache mix, GPU, CPU and PCIe perfectly overlapped | 107.7 t/s |
| **measured (S10)** | **35.2 t/s = 60 % of the serial ceiling** |

Serial parts per token: GPU dense + LM head 5.63 ms, cache hits 3.66 ms, CPU misses 4.37 ms, uploads 3.46 ms =
17.1 ms; measured 28.4 ms. The ~11 ms gap is the 48 per-layer sync/plan round trips plus quantized GEMV kernels
below copy bandwidth. More decode needs overlapping the CPU misses and uploads with GPU work (ceiling 108), or a
higher hit rate.

## Prefill ceilings (4783-token prompt as benchmarked, prefetch path = full uploads)

Every ubatch uploads the full 26.58 GiB, including the last, partial one: a 4783-token prompt at -ub 2048 is
3 ubatches = 3 full uploads.

| -ub | ubatches | upload | compute | serial ceiling | overlapped ceiling | measured |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 10 | 10.85 s | 2.10 s | 369 | 441 | 503 * |
| 1024 | 5 | 5.43 s | 2.10 s | 636 | 882 | 802 (S11) = 91 % |
| 2048 | 3 | 3.26 s | 2.10 s | 894 | 1469 | 932 (S12) = 63 % |

\* Measured without prefetch: the MoE cache supplies its ~25 % of the experts from GPU memory, so fewer bytes
cross PCIe (ceiling with that ~590). Compute-only ceiling (no uploads, infinite ubatch): 2282 t/s.

Takeaways:
- -ub 1024 is close to its ceiling: little left there.
- -ub 2048 has the largest gap (932 of 1469): upload and compute overlap only partly. Better overlap there is
  worth up to +58 %.
- The number of ubatches, not the prompt length, sets the upload cost: a partial last ubatch costs a full upload.
- Prefetch uploads experts the cache already holds; skipping them saves up to ~25 % of the bytes.
