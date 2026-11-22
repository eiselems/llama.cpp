# Hill-climb log (inference-optimisation method)

Method: github.com/sybil-solutions/inference-optimisation-skill. Ceilings: [THEORY.md](THEORY.md).
Goal for 2026-11-22 (until 15:00): climb prefill and decode on the presets S10 / S11 / S12.

## Harness (from 2026-11-22 11:15)

- `scripts/inkling/bench.py` (vendored, MIT, local changes marked `inkling:`), `queue.sh` (serialized, lock),
  `serve.sh` (start/stop a preset by PID), `vram.sh` (whole-GPU memory probe, 7.5 GiB cap = 500 MiB headroom).
- Per run: startup, greedy correctness gate (2+3, summary, loop check), c1 decode = median of 6 prompts
  (code, explanation, story, comparison table, bash, summary; 512 tokens; model card sampling temp 1.0,
  top-p 1.0, top-k 0, min-p 0), cold prefill on unique documents of ~1k and ~4.8k tokens, long-context needle
  run (~30k tokens, 3 codes, all must be found), peak VRAM.
- Experiment dir: `~/eval-runs-20261122-inkling/climb/` (configs, ledger.jsonl, logs); copies of the configs and
  the ledger are in `docs/inkling-perf/climb/`.
- Acceptance: a candidate must beat both neighbouring controls on its target metric, regress nothing by >3 %,
  and pass the gate and the needle check.

Note: these numbers are not directly comparable with the S0-S12 rows of the old `inkling-bench.sh` (one repeated
prompt, 128 tokens, single sample). Rows from the new harness are marked "(bench.py)" in the chart.

## Log

| time | experiment | result | verdict |
|---|---|---|---|
| 11:17 | harness dry run (S10, 2 reps) | gate PASS, c1 36.0, prefill 655 (914 tok) / 753 (3879 tok), needle 3/3 at 15.7k | harness works; prompt sizes corrected |
| 11:20-11:33 | baseline, interleaved S10-a, S11-a, S10-b, S12-a, S10-c (6 decode samples each) | see table below; all gates PASS, needles 3/3 at 29,725 tokens | new baseline |

### Baseline (bench.py, 2026-11-22 11:33)

| preset | c1 decode (median of 6) | prefill 958 tok | prefill 4,724 tok | needle run: prefill 29.7k / decode after | VRAM peak |
|---|---:|---:|---:|---:|---:|
| S10 (3 runs) | 37.3 / 36.9 / 36.7 -> **36.9** | 688 | 735 | 682 / 27.4 | 7.35 GiB |
| S11 | 35.3 | 745 | 820 | 762 / 26.9 | 7.25 GiB |
| S12 | 33.1 | 737 | **950** | 870 / 27.4 | 7.13 GiB |

Noise: the three S10 runs spread 1.6 % on decode and 0.5 % on prefill, well inside the 3 % acceptance rule.
The S10 > S11 > S12 decode order and the S12 > S11 > S10 prefill order from the old harness hold.
(The runs were flagged ON_BATTERY by mistake: the wireless mouse battery reports online=0; fixed in bench.py.)

### Step 1: where decode time goes (nsys, 2026-11-22 11:45)

S10, one 256-token decode request after a short prompt, captured with `nsys launch/start/stop` (request window
only); second capture with `--cuda-graph-trace=node` for per-kernel times. 34.5 t/s under the profiler.

| per token | measured | serial ceiling (THEORY.md) |
|---|---:|---:|
| GPU kernels | 16.4-17.1 ms | ~9.3 ms (dense + LM head + cache hits at 405 GB/s) |
| - dense GEMV (Q8_0 `mul_mat_vec_q<8>`, ~290 calls) | 5.6 ms | 4.4 ms -> ~79 % of bandwidth |
| - expert GEMV from the cache (NVFP4 `mul_mat_vec_q<40>`, ~87 calls) | 4.2 ms | 3.7 ms -> ~86 % |
| - ~2,970 small kernels (norms, attention, glue, activation quantize, ...) | ~7.3 ms | ~0 |
| expert uploads (H2D, 276 copies, 86 MB) | 3.6 ms | 3.5 ms |
| GPU idle (CPU computes ~70 % of the misses, host plans, syncs) | 8.8 ms | ~4.4 ms of CPU expert work |
| total | 29.0 ms | 17.1 ms |

Host: 668 `cudaStreamSynchronize` per token (~14 per layer application), 96 graph launches per token (the
decode graph is split into ~96 pieces by the CPU/GPU handovers), 3,348 kernels per token.

Where the ~11 ms above the serial ceiling go: ~7 ms is per-kernel overhead of ~3,000 small kernels, ~4 ms is GPU
idle time beyond the CPU expert work (handovers, syncs). The two big GEMV groups are already at 79-86 % of
bandwidth. Ranked decode levers from this: (a) fewer CPU/GPU handovers and syncs per layer, (b) fewer, larger
kernels (fusion), (c) GEMV efficiency.

| 11:40-11:50 | E1: large batches skip reading the router ids back before the expert upload (no host sync per MoE layer), opt-in LLAMA_EXPERT_COPY_NOSYNC=1, S12; ctl / cand / ctl | prefill 4.7k 954.4 / 952.8 / 953.0, 958 tok 745 / 743 / 743, c1 33.2 / 33.8 / 33.6; gate PASS, needles 3/3 | **rejected**: neutral (with prefetch the weights never take that path; the scales are tiny) |

### Step 2: prefill at -ub 2048 is compute bound, not upload bound (nsys, 11:50)

S12, one 4,777-token prompt (3 ubatches), 935 t/s under the profiler:

| | time |
|---|---:|
| window | 5.11 s |
| GPU compute (kernels) | 3.97 s |
| uploads (H2D, copy stream, 85.6 GB at 26.2 GB/s) | 3.34 s, of which **2.69 s overlap compute** |
| GPU fully idle | 0.30 s |

The prefetch overlap works. The limit is compute: 3.97 s = 11.6 TFLOPS effective (~26 % of FP16 peak), about
2x the 50 %-of-peak assumption in THEORY.md. Kernels: `mul_mat_q` (experts + dense) 1.39 s (36 %),
**`flash_attn_tile` 1.22 s (31 %)**, the rest ~1.3 s.

Attention runs in the generic tile kernel because PR #32740 adds the indexed relative-position bias only there
(the vec and MMA kernels ignore the bias). In the PR thread realugbun reports an unpublished MMA kernel with the bias
(biased mask load per KV tile, ncols2 forced to 1): 8K prefill 1300 -> 1527, 32K 838 -> 1149 on a 5060 Ti.
Decode also moved from the vec kernel to the tile kernel (0.4 ms/token here).

Consequence for the levers: at -ub 2048 better upload overlap has little left (0.65 s of upload not overlapped);
attention in an MMA kernel is the biggest prefill lever there. At -ub 1024 (S11) prefill is still upload bound,
so skipping experts that are already in the MoE cache (step 4) should help there.

| 11:50-12:10 | E2 (step 4): prefetch skips experts already in the MoE cache, opt-in LLAMA_EXPERT_PREFETCH_SKIP=1, S11; ctl / skip / ctl / skip / ctl | prefill 958 tok 745.8 / **859.5** / 744.7 / **857.5** / 745.8 (+15 %), 4,724 tok 819.8 / **913.2** / 819.8 / **912.9** / 820.1 (+11 %); c1 decode 35.7 / 35.8 / 35.7 / 35.9 / 35.2; decode after the 29.7k needle prompt 29.0 / **27.0** / 28.9 / **27.3** / 28.8 (-6 %, reproducible); gate PASS, needles 3/3 | **rejected for now** (3 % rule): big prefill win, but decode after long prompts drops 6 % for a reason not found yet (skip mode does not touch decode; VRAM peak identical). Branch `inkling-try-prefetch-skip`. Worth a follow-up |

| 12:05-12:20 | E3 (step 3): MoE cache looks the CPU experts up on the CPU, before the cached experts (`LLAMA_MOE_CACHE_CPU_IDS_ON_CPU=1`), S10; ctl / cand / ctl | c1 37.08 / 36.36 / 36.67, prefill 688 / 698 / 687 and 735 / 736 / 735, after 29.7k 28.2 / 27.6 / 27.5; gate PASS, needles 3/3 | **rejected**: neutral. nsys: the CPU part now overlaps the cached-expert graph (gap 200 -> ~50 us per layer), but the critical path is attention graph (~175 us) -> host waits for the router ids -> upload of the missing experts (~180 us, GPU idle) -> cached-expert graph (~105 us). Branch `inkling-try-cpu-ids` |
| 12:20-12:30 | E3b: E3 with CPU share 0.85 / 0.95 (fewer uploads now that the CPU part overlaps), S10; ctl / 0.85 / 0.95 / ctl | c1 36.8 / 37.2 / 36.6 / 37.1; gate PASS, needles 3/3 | **rejected**: neutral |
| 12:30-12:50 | E4 (step 3): MoE cache background fill (`LLAMA_MOE_CACHE_ASYNC_FILL=1`): decode misses run on the CPU for the token that missed them, the slot is filled on a second stream and used from the next token on; S10; ctl / E4 / E4+E3 / ctl / E4 / ctl | c1 36.95 / **37.70** / 36.99 / 37.19 / **37.97** / 37.09; prefill unchanged; after 29.7k 27.8 / 28.0 / 27.9 / 29.4 / 29.9 / 27.4; gate PASS, needles 3/3; 98 % of the misses on the CPU, ~8,900 background fills per two 256-token requests, hit rate 84.4 % | **accepted** (+2.1 %, beats every neighbouring control). Merged as d0c94745, preset S13 = S10 + E4. Combined with E3 it is neutral |
| 12:50-13:00 | E2 follow-up: paired long-prompt test (29.7k needle prompt, then 256 tokens), S11 control / skip, twice | decode after the long prompt 26.90 / 26.85 and 29.14 / 29.22; prefill 29.7k 757 / 789 and 759 / 789 | the -6 % was noise: the metric is one sample and swings between ~26 and ~29 in both modes |
| 13:05-13:18 | E2 repeated on main (with E4): S11 ctl / skip / ctl / skip / ctl | prefill 958 tok 744 / **857** / 743 / **859** / 748, 4,724 tok 819 / **912** / 820 / **913** / 820, 29.7k 761 / **793** / 761 / **793** / 761; c1 35.3 / 36.0 / 35.7 / 35.8 / 35.8; gate PASS, needles 3/3 | **accepted** (+15 % / +11 % / +4 % prefill). Merged as 1bf03dedd, preset S14 = S11 + E2 |
| 13:25 | nsys S13 (E4) decode | uploads now on the fill stream (78.5 MB/token) in parallel with the device; GPU idle 9.4 ms/token: the CPU part (~98 % of the misses) is the new critical path; per miss CPU compute (~160 us) costs about what the upload did (~165 us), hence only +2 % | next: CPU and PCIe paths in parallel (E5) |
| 13:12-13:25 | threads on S13: -t 6 / 8 / 12; ctl / t8 / t12 / ctl | c1 37.35 / 36.49 / 35.52 / 37.56; after 29.7k 30.0 / 27.4 / 19.7 / 28.0 | **rejected**: more threads are slower (the CPU part is RAM bound; SMT siblings contend) |
| 13:28-13:40 | E5: this token's uploads over the fill stream (device stream waits on an event, host does not) + E3, so CPU part and PCIe run in parallel; ctl(S13) / CPU share 0.7 / 0.5 / ctl(S13) | c1 37.53 / 36.11 / 35.06 / 37.51; gate PASS, needles 3/3 | **rejected**: slower. nsys: the host still blocks ~345 us per layer, because the scheduler queues the GPU->CPU copy of the CPU part's input behind the event wait; share 0.5 doubles the uploaded bytes on top. Needs a scheduler change (copy the CPU split's inputs before the expert phase / on another stream). Branch `inkling-try-split-path` |
| 13:40-14:00 | E2 on S12 (-ub 2048): ctl / skip / ctl / skip / ctl | prefill 958 tok 739 / **852** / 743 / **848** / 742, 4,724 tok 945 / **1,003** / 954 / **1,004** / 955 (+5.6 %), 29.7k 873 / 872 / 875 / 877 / 877; c1 33.7 / 33.3 / 33.5 / 33.1 / 33.7 (-1.3 %); gate PASS, needles 3/3 | **accepted**: preset S15 = S12 + E2, first preset above 1,000 t/s prefill (code already on main) |
| 13:55 | nsys E4+E3 decode | activations reach the CPU early (15 us sync), uploads run on the fill stream in parallel, but in heavy layers the CPU computes its misses for ~500 us alone (GPU idle); averaged ~250-280 us per layer for ~5.3 MB = ~20 GB/s. Without the cache (all experts on the CPU, larger batches per call) the CPU kernel reaches ~33 GB/s | the CPU expert part is the decode critical path; it does not overlap the next attention (data dependency) |
| 14:00-14:12 | thread polling on S13: ctl / --poll 100 / --poll 100 --cpu-strict 1 / ctl | c1 37.19 / 37.48 / 37.52 / 37.15 | **rejected**: +0.8 %, inside noise; thread wake-up is not the cost |
| 14:13-14:25 | S13 at -ub 512 with the freed compute buffer in the MoE cache: ctl / cache 3850 / cache 3950 / ctl | c1 37.33 / 38.08 / **38.39** / 37.08; prefill 687 / 367 / 367 / 688 and 736 / 379 / 379 / 736; VRAM peak 7.35 / 7.38 / 7.48 / 7.35 GiB; gate PASS, needles 3/3 | trade-off, not a replacement for S13 (prefill -50 %): kept as an explicit "max decode" preset S16 (+3 % decode) |
| 14:27-14:38 | S16 tweaks: ctl / -t 5 / --poll 100 / ctl | c1 38.86 / 37.77 / 38.19 / 38.36 | **rejected**: slower / neutral. S16 itself confirmed at ~38.6 |

## STATUS (2026-11-22 14:40)

All numbers: bench.py, median of 6 decode prompts, gate PASS and needles 3/3 at 29.7k on every row.

| metric | upstream master (S0, old harness) | bench.py baseline 11:33 | now (best preset) | ceiling (THEORY.md) |
|---|---:|---:|---:|---:|
| prefill, 4.7k-token prompt | 56 | 950 (S12) | **1,003 (S15)** | ~1,200 realistic at -ub 2048 (compute bound, 26 % of FP16 peak); 1,469 with 50 % of peak |
| prefill, 1k-token prompt | 58 | 745 (S11) | **858 (S14)** | upload bound at -ub 1024 |
| decode, short prompt | 18 | 36.9 (S10) | **38.6 (S16)** / 37.8 (S13, full prefill) | 58 serial, 108 fully overlapped |

Accepted today: E4 background cache fill (+2 % decode, S13), E2 prefetch skips cached experts (+15 % / +11 % prefill on
S11 = S14, +5.6 % on S12 = S15, first preset above 1,000 t/s), S16 max-decode preset (-ub 512, bigger cache,
+3 % decode for half the prefill). Rejected (documented above and in the README): E1, E3, E3b, E5, thread counts,
polling.

What limits further gains (measured with nsys):
- Decode (~26 ms/token at S13): per layer application attention graph ~190 us (the host must wait for the router ids),
  cached experts ~110 us, then the CPU computes its misses (98 % of the misses since E4) for ~250-500 us while the GPU
  idles. The CPU expert kernel reaches only ~20 GB/s at one token (~1.25 experts per call) against 48 GB/s RAM and
  ~33 GB/s at larger batches. Plus ~7 ms/token of ~3,000 small GPU kernels.
- Overlapping the CPU part with the GPU is blocked by the scheduler: inputs of the CPU split are copied on the device
  stream behind the expert uploads (E5); without uploads on the device stream (E4+E3) there is little GPU work left to
  overlap with (the next attention depends on the CPU result).
- Prefill at -ub 2048: compute bound; flash attention runs in the generic tile kernel (31 % of prefill) because the
  indexed relative-position bias exists only there.

Ranked next levers:
1. CPU MoE kernel at one token (NVFP4 MUL_MAT_ID, ~20 -> ~40 GB/s): decode ~38 -> ~43 t/s. Kernel work.
2. Relative-position bias in the MMA flash-attention kernel (realugbun measured 838 -> 1149 at 32K): prefill
   ~1,000 -> ~1,150 at 4.7k, more at long context. Kernel work.
3. Fewer small GPU kernels per token (fuse activation quantize into the GEMV, more glue fusion): ~2-3 ms/token.
4. Scheduler: copy the CPU split's inputs before the expert phase / on another stream, then retry E5 (CPU + PCIe in
   parallel for the misses).
5. Bias support in the vec kernel for decode attention (0.4 ms/token).

### After the STATUS: next levers (2026-11-22 from 14:45)

| time | experiment | result | verdict |
|---|---|---|---|
| 14:45 | microbenchmark of the CPU NVFP4 x Q8_0 kernel (AVX2), one expert matrix | 10.4 GB/s per thread, ~61 GB/s on 6 threads: the kernel is not the decode limit; a SIMD scale variant was slower (50 GB/s) | lever 1 as stated is wrong |
| 14:50 | nsys CPU sampling of decode (S16) | OpenMP worker threads: ~62 % in `gomp_barrier_wait_end`, ~15 % in the NVFP4 kernel; main thread mostly waits in the CUDA driver. The CPU expert part costs ~4-5 ms/token in total; the GPU 16.5 ms | the bulk is GPU time, ~7 ms of it in ~3,000 small kernels |
| 14:55-15:05 | OpenMP wait policy on S13: ctl / OMP_WAIT_POLICY=ACTIVE / GOMP_SPINCOUNT=INFINITE / ctl | c1 37.69 / 37.70 / 37.34 / 37.17 | **rejected**: neutral (the barrier time is idle waiting, not overhead) |
| 15:15-15:40 | E6: indexed bias in the vec FA kernel for decode (`GGML_CUDA_FATTN_BIAS_VEC=1`), S13; ctl / vec / ctl / vec / ctl | c1 37.66 / 37.16 / 37.78 / 37.41 / 37.77; test-flash-attn-indexed-bias passes (max error 4.7e-09 vs 3.5e-05 for tile) | **rejected** (-1 %): nsys vec 0.78 ms/token vs tile 0.41 + dequantize 0.28 ms; with GQA 3 the vec kernel decodes the q4_0 K/V once per Q head. Branch `inkling-try-fattn-vec-bias` |
| 15:45-16:00 | E7: indexed bias in the MMA FA kernel for prefill (`GGML_CUDA_FATTN_BIAS_MMA=1`), S15; ctl / mma / ctl / mma / ctl | prefill 29.7k 877 / **903** / 872 / **903** / 875 (+3.2 %); 958 tok 853 / 844 / 846 / 845 / 851, 4,724 tok 994 / 996 / 1,000 / 997 / 1,002 (neutral); c1 33.4 / 33.7 / 33.4 / 33.5 / 33.6; gate PASS, needles 3/3; test-flash-attn-indexed-bias passes with lower errors | **accepted** (small): merged a39686409, preset S17 = S15 + E7. nsys: attention 1.22 s on both kernels for a 4.8k prompt: the kernels are bound by gathering the per-element int32 bias indices (re-read by all 24 Q heads), not by the math |

Updated lever list (16:05):
1. Bias indices: derive the bin from the query/key positions in the kernel, or share the indices across the
   heads of a block (GQA grouping), instead of an int32 per (query, key) read by every head. Attention is 31 % of
   prefill at -ub 2048 and dominated by this; also helps the decode attention (E6 lost to it as well).
2. Fewer small GPU kernels per decode token (~7 ms/token in ~3,000 kernels).
3. Scheduler: copy the CPU split's inputs before the expert phase, then retry E5.
