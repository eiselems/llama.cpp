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
