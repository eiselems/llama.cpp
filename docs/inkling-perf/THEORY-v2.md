# Theoretical ceilings v2 (2026-11-27, S32 on IQ4_NL)

Re-measurement of [THEORY.md](THEORY.md) (2026-11-22, S10-S15 on UD-Q4_K_M) on the current best 32k preset S32
(`scripts/inkling/serve.sh start S32`, auto MoE cache = 3,161 MiB, ~820 MiB desktop). Method as in PLAN.md step 1:
nsys `launch` + `start`/`stop` around one request; decode = 256-token reply after a short prompt, model card sampling
(T 1.0, top-k 0); prefill = bench.py `document()` prompts of 4,724 and 28,986 tokens.

## Hardware (re-measured, unchanged)

| what | 2026-11-22 | 2026-11-27 |
|---|---:|---:|
| PCIe upload, pinned, 1 GiB | 26.3 GB/s | 26.30 GB/s (gen 4 x16 under load; idles at gen 1) |
| PCIe download, pinned | - | 26.09 GB/s |
| GPU memory, D2D copy | 404.6 GB/s | 397 GB/s (desktop at 12 % GPU) |
| CPU RAM read, 6 threads | 48.6 GB/s | 48.1 GB/s (1 thread 45.3, 3 threads 51.4 = best, 12 threads 45.8) |

## Model bytes per decode token (IQ4_NL)

| | v1 (Q4_K_M) | v2 (IQ4_NL) |
|---|---:|---:|
| dense weights read (loops x3) | 1.66 GiB | 1.66 GiB |
| LM head | 0.46 GiB | 0.46 GiB |
| active experts | 1.66 GiB | 1.72 GiB (expert 4.27 MiB; layers 20-23 Q8_0, 8.07 MiB) |

## Decode: cache (trace replay, 3,275 tokens; simulator 76.6 % vs server 77.1 %)

Hit rate 77 % (v1 assumed 83 %). Per token: 62.8 experts on the CPU (290.6 MB), 26.9 uploaded (124.6 MB).

| group | hit % | CPU misses/tok | same expert as previous token | selected in an earlier pass of this token |
|---|---:|---:|---:|---:|
| layers 0-5 | 52.3 | 15.3 | 27 % | - |
| loop pass 1 (6-17) | 82.3 | 11.4 | 48 % | - |
| loop pass 2 | 83.3 | 10.8 | 43 % | 62 % |
| loop pass 3 | 77.6 | 14.4 | 43 % | 61 % |
| layers 18-23 | 74.1 | 8.3 | 48 % | - |

Cache size (simulated): 3,161 MiB 75.9 % / 3,700 79.5 % / 4,500 83.6 % / 6,000 89.2 %; each +540 MiB is about -9 CPU
misses (-43 MB) and -19 MB of uploads per token.

## Decode: where a token goes (39.9 t/s unprofiled = 25.1 ms; 26.95 ms under nsys)

| part | ms/token | note |
|---|---:|---|
| GEMVs: dense ~4.2, cached experts 4.73, LM head 1.19 | 10.1 | 420 / 330 / 414 GB/s: at bandwidth |
| ~2,250 small kernels (< 10 us, of 2,700 per token) | 5.35 | ctrl_down/out 1.0, norms, conv state, mix, clip, quantize, ... |
| expert uploads on the device stream | 4.87 | S32 has no `LLAMA_MOE_CACHE_ASYNC_FILL`; 125 MB at PCIe speed, serial with compute |
| GPU idle: CPU experts not overlapped | 2.04 | |
| GPU idle: CUDA driver (96 graph launches, 175 syncs) | 2.85 | |
| GPU idle: sampling (`llama_sampler_dist_apply`, 201k vocab, top-k 0) | 1.10 | on the critical path |
| GPU idle: other host work | ~0.6 | |

CPU experts: ~7.5 ms/token per thread for 290.6 MB = ~39 GB/s (v1 said ~20 GB/s on NVFP4; K7's microbench 46-52),
~75 % of it overlapped with GPU work.

| decode ceiling | v1 | v2 | S32 measured (39.9) |
|---|---:|---:|---:|
| all parts serial (GPU 9.45 + PCIe 4.75 + CPU 6.04 ms) | 58.4 | 49.4 | 81 % |
| current structure without overhead (GPU stream: 9.45 + 4.75 ms, CPU hidden) | - | 70.4 | 57 % |
| all experts in VRAM | 99.6 | 98 | 41 % |
| GPU, CPU, PCIe perfectly overlapped | 107.7 | 106 | 38 % |

## Prefill (S32, -ub 2048, single staging slot)

| | 4,724 tok (3 ubatches) | 28,986 tok (15 ubatches) |
|---|---:|---:|
| measured, unprofiled | 1,251 t/s (cold first run 1,147) | 1,325 t/s |
| window under nsys | 3.64 s (1,299 t/s) | 21.76 s (1,332 t/s) |
| GPU compute | 2.48 s | 16.11 s |
| - mul_mat_q / flash attention / sgemm | 0.99 / 0.45 / 0.17 s | 4.97 / 5.20 / 0.93 s |
| D2D copies on the device stream (staging and cache -> compute buffer) | 0.475 s, 82.6 GiB | 2.375 s, 412.9 GiB |
| H2D uploads (prefetch stream) | 2.58 s, 63.8 GiB = 21.3 GiB/ubatch | 12.96 s, 319.9 GiB = 21.3 GiB/ubatch |
| uploads overlapped with compute | 1.94 s | 10.12 s |
| real GPU gaps | 0.04 s | small |
| ceiling: device stream as now (compute + D2D) | 1,596 t/s (78 %) | 1,568 t/s (85 %) |
| ceiling: no D2D, perfect overlap = max(compute, uploads) | 1,831 t/s (68 %) | 1,799 t/s (74 %) |

v1 assumed 26.58 GiB uploaded per ubatch; prefetch-skip makes it 21.3. 4.7k is now upload bound (uploads 2.58 s >
compute 2.48 s); 29k is compute bound, with flash attention at 32 % and growing with the prompt.

## Levers this points at (status 2026-11-28, results in [README.md](README.md))

Decode:
1. Uploads off the device stream (4.87 ms): try S32 + `LLAMA_MOE_CACHE_ASYNC_FILL=1` (+2 % on S13 in E4; all misses
   then go to the CPU, about +27 per token). **Rejected, -4.2 %.** Overlapping this token's uploads with the cached
   experts (split-fill, O1) **rejected, -2.5 %**: the upload layers are already CPU bound.
2. Sampling (1.10 ms, ~4 %): a faster full-vocabulary dist sampler, or overlap it with the next token's work.
   **Done: `--backend-sampling`, +3.6 % (S36).**
3. Small kernels (5.35 ms) and driver gaps (2.85 ms): fusion and fewer graph splits/syncs. Graph launches: ~16 us
   each, 96 per token (1.5 ms); graphs off -7.5 %, eager first nodes -1.5 % (rejected). **K8 router norm fusion
   +0.5 % (noise)**; the fusion candidates left are ~0.3-1 % each. Fewer splits/syncs: blocked by the per-layer host
   round trip the CPU experts need. A GPU-driven loop (prototype) and zero copy of the small tensors (+0.9 %) do
   not help: the per-layer CPU expert work, not the orchestration, sets the pace.
4. Layers 0-5 hit only 52 % with little reuse between tokens (27 %): a cache policy question (simulator).
   **Rejected (O2, -1.3 %)**: a layer costs max(upload, CPU), not the sum, and with that cost the uniform CPU share
   0.7 is already best (`scripts/inkling/moe-budget-sim.py`). Left: a larger cache or smaller experts.

Prefill:
1. D2D copies (11-13 %): let the matmuls read the staging slot / cache slots in place instead of copying.
   **Done with two slots (S38, prefill first): +6.0 % at 4.7k, +6.4 % at 25.7k; with one slot not possible.**
   **S39: the copies are the slots, compute buffer 1061 -> 662 MiB: two slots at no cost to the MoE cache; vs S37
   prefill +13.5 % at 4.7k, +11.4 % at 25.7k, decode +1.6 %.**
2. 4.7k: 0.64 s of the device stream waiting for uploads with one staging slot.
3. 29k: flash attention (5.2 s). **K10 (S37): the global layers' far K/V blocks use one bias value per row, -36 % of
   their attention time at 30k, +4.0 % prefill at 25.7k, exact.**
