# Inkling-2-Mini perf log (fork branch `inkling-2-perf-improvement`)

RTX 3070 8 GB, Ryzen 5 5600X, 32 GB RAM. Unsloth `Inkling-2-Mini-UD-Q4_K_M`, 32k ctx, experts in RAM.
Rows S0-S12: `scripts/inkling-bench.sh` (one repeated prompt, 128 tokens, single sample). Rows from B10 on:
`scripts/inkling/bench.py` (vendored from sybil-solutions/inference-optimisation-skill; median of 6 decode prompts,
greedy correctness gate, long-context needle check, control/candidate/control interleaving). Only rows that pass the
checks are recorded. Full experiment log and status: [PLAN.md](PLAN.md); ceilings: [THEORY.md](THEORY.md), re-measured on S32 in
[THEORY-v2.md](THEORY-v2.md).

## Decode fusion, upload overlap, per-layer cache budget (2026-11-28, 32k, on S36)

No new best; S36 stays the 32k preset. The decode ceilings were re-measured with nsys first (2026-11-27): hit rate
77 %, ~63 CPU misses per token, uploads in ~21 of the 48 layer applications, 96 graph launches per token.

| step | change (env) | result | exact? |
|---|---|---|---|
| K8 | topk-moe also fuses the router weight norm when it is `sum + eps` (SCALE) instead of a CLAMP, as in Inkling (`GGML_CUDA_TOPK_MOE_NORM_EPS=1`) | decode 43.04 vs 42.83 (3 pairs bench.py: 43.31 / 43.02 / 42.79 vs 42.95 / 42.59 / 42.95), **+0.5 %, inside noise**; not in a preset | gate PASS |
| K9 | one zero-sinks fill shared by all layers instead of one per layer (graph cleanup) | not measured on its own (one tiny kernel per layer) | yes |
| O1 | split-fill overlap: in decode, this token's expert uploads run on the fill stream while the cached experts compute; the uploaded experts run after the CPU part, in the same device split (no extra graph launch). mmvq treats id -1 as a zero row (`LLAMA_MOE_CACHE_SPLIT_FILL=1`, implies `LLAMA_MOE_CACHE_NEG_IDS=1`; branch `inkling-try-split-fill`, d028b9f88) | **rejected, -2.5 %**: decode 43.27 vs 44.40 / 44.46 (CPU share 0.7); share 0.6 / 0.5 / 0.3: 42.79 / 42.95 / 39.79 (control at 0.5: 41.67). `NEG_IDS` alone: 44.27 (neutral) | yes (greedy text identical) |
| O2 | per-layer CPU share: layers 0-5 (hit 52 %, little reuse) upload less, so their uploads stop evicting experts the loop layers reuse (`LLAMA_MOE_CACHE_CPU_SHARE_HEAD=6:0.95`; branch `inkling-try-cpu-share-head`, 166a57d9c) | **rejected, -1.3 %**: decode 43.55 vs 44.13 (3 pairs bench.py: 43.14 / 43.88 / 43.64 vs 44.47 / 43.77 / 44.16), prefill the same | gate PASS |

O1 numbers: median of 3 decodes of 512 tokens (model card sampling), not bench.py, so not comparable to the chart.
A microbench (real MUL_MAT_ID, IQ4_NL, Inkling shapes) had predicted +4-6 %: an upload on a second stream slows the
cached-expert GEMVs by only ~6 %. In the server it lost, because (nsys):

- misses come in clusters: a layer that uploads one expert also has ~200 us of CPU experts (~500 us with 2+ uploads),
  so the CPU part is already the critical path there and the overlap saved ~30 us instead of ~60;
- the second set of expert kernels runs in every layer, also with nothing uploaded, and costs 15-25 us on the ~28 of
  48 layer applications per token without uploads;
- a lower CPU share moves the work back onto PCIe.

Decode is now bound by the CPU experts in miss-heavy layers, and the CPU kernel already runs at RAM speed (K7).

O2 came from a trace replay (`scripts/inkling/moe-budget-sim.py`, S32 decode trace, 3,275 tokens; the unchanged
policy gives 75.9 % hit, as measured). It ranked share 0.95 for layers 0-5 best (-0.68 ms of misses per token), but
it added upload and CPU time. In the server the uploads of a layer run while its CPU experts compute, so a layer
costs about max(upload, CPU). In layers 0-5 the CPU part (~2.6 experts, ~245 us) already hides the ~1 upload
(~195 us), and moving it to the CPU only makes the CPU part longer. With the max cost (the `max` column) the uniform
share 0.7 is already best (7.73 ms per token; 0.95 for layers 0-5: 8.15, the A/B direction). The only gain left is
evicting layers 0-5 earlier (score weight 8: 7.58 ms, ~+0.6 %, inside noise; not built). The cache policy is close to
optimal for this cache size; the levers left are a larger cache (more VRAM) or smaller experts (lower-bit quant,
needs the quality evals).

## Sampling on the GPU (2026-11-27, 32k, bench.py, 2 pairs interleaved)

**New best at 32k: S36** = S32 + `--backend-sampling` (`scripts/inkling/serve.sh start S36`): decode **42.48 / 42.67**
t/s vs S32 41.14 / 41.06 (**+3.6 %**), prefill the same (958 / 4,724 tok: ~820 / ~1,268 t/s), gate PASS. nsys on S32
showed the CPU `dist` sampler over the 201k vocabulary at 1.1 ms per token with the GPU idle. With the model card
sampling (top-k 0, top-p 1, min-p 0, temp 1) every sampler of the chain is empty or runs on the backend. The server
falls back to the CPU for grammar (tool calls with a grammar), a reasoning budget and `n_probs`; seeds draw other
tokens than on the CPU.

Rejected the same day: S32 + `LLAMA_MOE_CACHE_ASYNC_FILL=1` (expert uploads off the device stream, all misses on the
CPU): decode 39.06 / 39.23 vs 40.98 / 40.71 (-4.2 %), the CPU part becomes the critical path.

## Kernel round K1-K7 (2026-11-24, 32k, bench.py, 3 runs interleaved with the previous preset each)

**New best at 32k: S32** = S27 + K1 + K2 + K3 + K4 + K6 (`scripts/inkling/serve.sh start S32`): prefill 958 / 4,724 /
29.7k tokens **844 / 1,291 / 1,394** t/s (S27: 809 / 1,036 / 1,063), decode **41.8** t/s (S27: 38.5), decode after a
29.7k prompt **39.7** (S27: 29.6). Every step except K2 is bit-identical (server greedy output, FA microbench `cmp`).

| step | change (env) | result vs previous | exact? |
|---|---|---|---|
| K1 = S28 | flash attention reads the bias table through its permuted view (`LLAMA_ATTN_BIAS_VIEW=1`): the gather was the prefill attention bottleneck, not the masked tiles; sliding-window layers skip their fully masked K/V blocks (`GGML_CUDA_FATTN_KV_WINDOW=1`) | pp5k **+12.9 %**, pp30k **+18.0 %** | yes |
| K2 = S29 | single-token decode attention straight from the q4_0 K/V cache, all GQA heads of a K/V head per block (`GGML_CUDA_FATTN_DEC_Q4=1`, fattn-dec-q4.cu) | decode after 30k **+30 %**, c1 +2.9 % | no: F32 instead of F16 math, KLD 0.0080 at -ub 1 (= any last-bit change), PPL ratio 1.003 +- 0.006 |
| K3 = S30 | the attention kernels subtract the per-query sinks from the table (`LLAMA_ATTN_BIAS_SINK_FOLD=1`), no SUB pass over 200 MB per layer | pp5k +2.2 %, pp30k +2.7 % | yes |
| K4 = S31 | fused expert SiTU in prefill (`GGML_CUDA_INKLING2_SITU_PLAIN=1`): the existing fusion needed a per-expert scale the IQ4_NL model lacks, 10 kernels -> 1 (590 -> 98 ms per 7k prompt) | pp5k **+5.6 %**, pp30k +5.8 % | yes |
| K5 | two single-token MUL_MATs of one input in one MMVQ launch (`GGML_CUDA_MMVQ_PAIR=1`, branch `inkling-try-k5-mmvq-pair`, 66529332b) | **rejected, 0 %**: correct, but fuses nothing here: the same-shape pairs (k + v, q + gate) are 12 / 52 nodes apart and the allocator reuses the partner's output memory in between; the adjacent control pairs are already one kernel. Upper bound: running the siblings concurrently (`GGML_CUDA_GRAPH_OPT=1`) gave 0 % (44.05/43.96 vs 43.86/44.01 t/s) | yes |
| K6 = S32 | reuse the q8_1 copy of a decode MUL_MAT input for the next MUL_MAT of that input (`GGML_CUDA_Q8_1_REUSE=1`): 452 -> 251 quantize launches per token | c1 +1.2 % | yes |
| K7 | CPU IQ4_NL expert kernel for one token: software prefetch of the weight rows (`GGML_CPU_MOE_PREFETCH=N`, branch `inkling-try-k7-cpu-moe`, 2f73f6f95), threadpool polling | **rejected**: the kernel already streams 46-52 GB/s with 6 threads (cold weights), the 48.6 GB/s RAM limit; prefetch makes it 3-5 % slower, 12 threads slower, `--poll 0/50/100` all noise. Decode per token: ~57 CPU experts (~255 MB, ~5 ms at RAM speed) + ~121 MB PCIe uploads, GPU busy 15.9 of 22.8 ms: the rest is per-layer orchestration (router sync, wake-up, result upload), not the kernel | yes |

Tools added: `GGML_CUDA_NODE_PROFILE=FILE` (per-node CUDA timing, found K4), FA microbench and CPU MoE microbench (in the
commit messages of K1/K2 and this section).

## Winners (2026-11-22 14:40, bench.py)

| goal | preset | prefill 958 / 4,724 tok | decode (median of 6) | VRAM peak | start |
|---|---|---:|---:|---:|---|
| **max prefill** | **S15** | 850 / **1,003** | 33.2 | 7.13 GiB | `scripts/inkling/serve.sh start S15` |
| **max prefill (IQ4_NL)** | **S19** | 890 / **1,080** (29.7k: **1,055**) | 36.7 | 7.49 GiB | `scripts/inkling/serve.sh start S19` |
| **max decode (IQ4_NL)** | **S20** | 503 / 578 (29.7k: 572) | **39.5** | 7.48 GiB | `scripts/inkling/serve.sh start S20` |
| **128k context (IQ4_NL)** | **S21** | 4k prompt: ~900 | 29.5 at 4k ctx, 34.5 short | 7.48 GiB (smoke) | `scripts/inkling/serve.sh start S21` |
| **32k, max prefill (since 2026-11-23)** | **S24** | 871 / **1,070** (29.7k: **1,073**) | 36.1 | 7.6 GiB | `scripts/inkling/serve.sh start S24` |
| **128k, decode first (since 2026-11-23)** | **S22** | 655 / 861 (94k: **760**) | **36.5** (after 94k: 18.4) | 7.6 GiB | `scripts/inkling/serve.sh start S22` |
| **128k, prefill first (since 2026-11-23)** | **S23** | **834 / 1,019** (94k: **831**) | 32.5 (after 94k: 18.8) | 7.6 GiB | `scripts/inkling/serve.sh start S23` |

IQ4_NL quality (card sampling, thinking): v1 16/16, v2 21/22, v3 10/10, v4 native 3/3, identical to UD-Q4_K_M.
S19/S20 are sized for 32k context (peak 7.49 GiB of 8). For 64k context lower `--moe-cache-mib` by ~400 MiB:
the eval at 64k with too big a cache ran out of VRAM.

**128k context, 2026-11-23** (bench.py `--long 120000`, ~790 MiB desktop, all runs gate PASS and 3/3 needles at 94k):
control S21c = S21 as before (cache 2336, half-life 128), median of 4 runs: 631 / 848 / 715 (94k) prefill, 33.7 decode, 18.2
after 94k. S22 = S21 + auto cache + half-life 64 + mask from positions + K/V chunks: decode **+8 %**, prefill +4 / +1.5 / +6 %.
S23 = S22 + prefetch (the staging fits again): prefill **+32 / +20 / +16 %**, decode -3 %. The compute buffer at 128k went
from 1,753 to 1,061 MiB, so the auto cache grew from 2,336 to 3,028 MiB (S22). Details: [PLAN.md](PLAN.md).

**Since 2026-11-23: `--moe-cache-mib auto`** (with `--moe-cache-margin-mib N`, default 256) sizes the cache from the free
VRAM after the KV cache, the compute buffers and the prefetch staging, so the presets no longer depend on the desktop's
GPU use. Desktop at ~790 MiB: S21 128k -> 2,336 MiB (fixed 2400 ran out of memory), S19 32k -> 2,782, 64k + prefetch -> 2,366;
speeds as the fixed presets, peak 7,822 MiB, runtime growth after startup ~160 MiB. The `~/startInkling@*.sh` launchers use it.
| **max prefill (new best)** | **S18** | 861 / **1,057** (29.7k: **1,039**) | 33.4 | 6.85 GiB | `scripts/inkling/serve.sh start S18` |
| **max prefill, long prompts** | **S17** | 845 / 996 (29.7k: **903**) | 33.6 | 7.13 GiB | `scripts/inkling/serve.sh start S17` |
| **max decode** (prefill halves) | **S16** | 367 / 379 | **38.6** | 7.48 GiB | `scripts/inkling/serve.sh start S16` |
| **balanced: decode first** | S13 | 689 / 734 | 37.8 | 7.35 GiB | `scripts/inkling/serve.sh start S13` |
| **balanced: prefill first** | S14 | 858 / 912 | 35.9 | 7.25 GiB | `scripts/inkling/serve.sh start S14` |

Start: upstream master on the same card did 57.6 / 56.4 t/s prefill and 18.0 t/s decode (S0).
All presets: q4_0 KV (PPL +0.24 %), 32k context, `--moe-cache-cpu-share 0.7`, `-t 6`; cache sizes assume a light
desktop (~560 MiB idle GPU use, see below). `serve.sh` holds the exact flags and environment variables.

![prefill and decode per step](progress.svg)

## Presets (2026-11-22 08:35)

VRAM peaks include the desktop. Check the idle GPU use first (`nvidia-smi`): it was ~1070 MiB on the evening of
2026-11-21 (heavy desktop) and ~560 MiB on the morning of 2026-11-22 (light desktop). The cache must shrink by about
the difference on a heavy desktop, or the server runs out of memory.

Common flags: `--fit off -ngl all -cmoe -b 2048 --moe-cache-cpu-share 0.7 -fa on -ctk q8_0 -ctv q8_0 -c 32768`

| preset | env + extra flags | -ub | cache MiB (light / heavy desktop) | prefill 1k / 4.8k | decode @1k / @4.8k | VRAM peak (light) |
|---|---|---:|---|---:|---:|---:|
| **max decode** (S16, bench.py; prefill halves) | `LLAMA_EXPERT_PREFETCH=3 LLAMA_MOE_CACHE_ASYNC_FILL=1`, `-ctk q4_0 -ctv q4_0` | 512 | 3950 / ~3450 | 367 / 379 (bench.py) | **38.4** (bench.py median) | 7660 |
| **max prefill, prefetch skip** (S15, bench.py) | `LLAMA_EXPERT_PREFETCH=1 LLAMA_EXPERT_PREFETCH_SKIP=1`, `-ctk q4_0 -ctv q4_0` | 2048 | 2650 / ~2150 | 850 / **1,003** (bench.py) | 33.2 (bench.py median) | 7301 |
| **balanced, prefetch skip** (S14, bench.py) | `LLAMA_EXPERT_PREFETCH=1 LLAMA_EXPERT_PREFETCH_SKIP=1`, `-ctk q4_0 -ctv q4_0` | 1024 | 3200 / ~2700 | **858 / 912** (bench.py) | 35.9 (bench.py median) | 7424 |
| **decode first, background fill** (S13, bench.py 2026-11-22) | `LLAMA_EXPERT_PREFETCH=3 LLAMA_MOE_CACHE_ASYNC_FILL=1`, `-ctk q4_0 -ctv q4_0` | 1024 | 3650 / ~3150 | 689 / 734 (bench.py) | **37.8** (bench.py median) | 7526 |
| **decode first** (S10) | `LLAMA_EXPERT_PREFETCH=3`, `-ctk q4_0 -ctv q4_0` | 1024 | 3650 / ~3150 | 638 / 724 | **35.2 / 33.0** | 7604 |
| **balanced** (S11) | `LLAMA_EXPERT_PREFETCH=1`, `-ctk q4_0 -ctv q4_0` | 1024 | 3200 / ~2700 | 682 / 802 | 33.4 / 31.6 | 7502 |
| **max prefill** (S12) | `LLAMA_EXPERT_PREFETCH=1`, `-ctk q4_0 -ctv q4_0` | 2048 | 2650 / ~2150 | 676 / **932** | 32.0 / 27.5 | 7382 |
| max prefill, q8_0 KV (S9) | `LLAMA_EXPERT_PREFETCH=1` | 2048 | 2400 / 2000 | 665 / 935 | 31.1 / 28.5 | 7504 |
| q8_0 KV, single slot (S8) | `LLAMA_EXPERT_PREFETCH=3` | 1024 | 3400 / 3000 | 632 / 724 | 33.9 / 31.8 | 7692 |
| no prefetch (S5) | - | 1024 | 3400 / 3000 | 582 / 686 | 33.6 / 32.0 | 7246 |

q4_0 KV quality check (wikitext-2, -c 4096, 6 chunks): PPL 5.7909 +/- 0.121 vs 5.7771 +/- 0.120 with q8_0 (+0.24 %,
within the error). It frees ~270 MiB at 32k context, which goes to the MoE cache.

Other measured option: `LLAMA_EXPERT_PREFETCH=3 -ub 2048 --moe-cache-mib 2800` (light desktop) = 632 / 908 prefill,
32.2 / 29.3 decode - nearly max prefill with better decode.

The same numbers as a table: [progress-table.md](progress-table.md) (raw: [progress.tsv](progress.tsv)).

## ubatch x cache sweep (CPU share 0.7, commit 1410a556)

| -ub | cache MiB | VRAM peak | pp 1k | pp 4.8k | tg @1k | tg @4.8k |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 3400 | 7534 | 453.8 | 502.5 | 34.1 | 32.8 |
| 1024 | 3000 | 7354 | 575.6 | 669.3 | 33.0 | 30.9 |
| **1024** | **3400** | 7758 | **582.4** | **685.8** | **34.1** | **32.1** |
| 2048 | 2400 | 7226 | 548.5 | 776.3 | 30.8 | 27.8 |
| 2048 | 2800 | 7626 | 568.5 | 785.1 | 32.0 | 29.3 |
| 1024 | 3400, share 0.6 / 0.8 | 7758 | 580.7 / 577.9 | 675.1 / 677.8 | 33.6 / 33.2 | 31.4 / 31.9 |
| 4096 | 1600 | 7808 | 522.1 | 730.0 | 26.4 | 23.5 |
| 4096 | 2000 | OOM (2.4 GiB compute buffer) | - | - | - | - |

-ub 4096 loses: the bigger compute buffer forces a small cache, and the saved uploads do not make up for it.

**Recommended everyday preset (since 2026-11-22 08:30, step S8):**
`LLAMA_EXPERT_PREFETCH=3` + `--fit off -ngl all -cmoe -b 2048 -ub 1024 --moe-cache-mib 3400 --moe-cache-cpu-share 0.7 -fa on -ctk q8_0 -ctv q8_0 -c 32768`
= 632.4 / 723.9 t/s prefill, 33.9 / 31.8 t/s decode (one ~445 MiB prefetch slot, full cache).

Previous everyday preset (S5, still valid without the env var):
`--fit off -ngl all -cmoe -b 2048 -ub 1024 --moe-cache-mib 3400 --moe-cache-cpu-share 0.7 -fa on -ctk q8_0 -ctv q8_0 -c 32768`

Prefill-heavy mode (merged, opt-in, step S6): `LLAMA_EXPERT_PREFETCH=1` with `--moe-cache-mib 2600` (the two
staging slots need ~790 MiB) = 672.9 / 798.5 t/s prefill, 31.4 / 29.8 t/s decode.
Max-prefill mode (step S7): `LLAMA_EXPERT_PREFETCH=1 -ub 2048 --moe-cache-mib 2000` = 659.8 / **924.9** t/s prefill,
29.2 / 26.7 t/s decode, VRAM peak 7620 MiB.
Middle option: `LLAMA_EXPERT_PREFETCH=1 -ub 1536 --moe-cache-mib 2400` = 615.0 / 852.7 t/s prefill, 30.1 / 28.4 t/s
decode, VRAM peak 7771 MiB. Prefetch with cache 2800 at -ub 1024 runs out of memory during the 4.8k prompt (7841 MiB).
Prefetch -ub 2048 with cache 2200 instead of 2000: 661.9 / 921.4 prefill but decode drops to 26.2 / 25.1 (VRAM peak 7815 MiB, too close to the limit); 2000 stays.
Balanced preset with `-t 5` instead of 6: 572.6 / 678.9 prefill, 33.3 / 31.4 decode - no better, `-t 6` stays.
Balanced preset at `-ub 1280` (cache 3400): out of memory during the speed runs (7841 MiB peak). About 7.8 GiB is the practical ceiling on this card with the desktop running.

## Tried and not kept

| what | result | why |
|---|---|---|
| PR #32752 (mask in chunks) | not applied | superseded by #32740 (bias inside flash attention) |
| PR #32076 (MoE cache without syncs) | rebased on 8ed3c071 (2026-11-22 08:52): no crash any more, output correct, but S10 settings give 628.9 / 718.3 prefill and 32.1 / 29.0 decode vs 637.5 / 723.9 and 35.2 / 33.0 without it (dedup off: 30.1) | slower on Linux CUDA; local branch `inkling-nosync-try` |
| -ub 4096 | 522 / 730 t/s prefill, 26.4 t/s decode at cache 1600; cache 2000 OOM | compute buffer too big for 8 GB |
| DSpark speculative decoding (PR #32218 + #32773), draft n 3 / 2, cache 2400 | 23.5 / 31.3 t/s decode vs 34.1 without; acceptance 44-55 % at T=1 | the draft needs ~0.5 GiB VRAM (smaller MoE cache), and 3-4 token verification batches select more experts (more misses); local branch `inkling-dspark-try` |

| gate/up-only prefetch, 1 slot (LLAMA_EXPERT_PREFETCH=2), cache 2900, branch `inkling-prefetch-half` | 516.8 / 569.7 prefill, 32.5 / 30.4 decode | worse prefill than no prefetch: the down uploads on the compute stream compete for PCIe |
| prefetch only the first half of each tensor (LLAMA_EXPERT_PREFETCH_FRAC=0.5), cache 3000, branch `inkling-prefetch-range` | 533.1 / 586.2 prefill, 32.9 / 28.2 decode | worse than S5 and S6: the directly uploaded rest competes with the prefetch stream; smaller slots need the chunk pipeline in NEXT.md |
| prefetch at -ub 2048 with the MoE cache off (are the cache syncs slowing prefill?) | 591.4 / 901.2 prefill, 18.4 decode vs 676 / 932 with cache 2650 (S12) | no: the cache helps prefill as well |

Hill-climb log with every experiment (2026-11-22 from 11:15, bench.py harness): [PLAN.md](PLAN.md).

Theoretical ceilings (measured hardware, model bytes/FLOPs): [THEORY.md](THEORY.md).| E1: large batches skip reading the router ids back before the expert upload (`LLAMA_EXPERT_COPY_NOSYNC=1`), S12, branch `inkling-try-copy-nosync` | prefill 4.7k 952.8 vs controls 954.4 / 953.0 | neutral: with prefetch the expert weights never take this path |
| E3: CPU-expert lookup on the CPU so the CPU part overlaps the cached experts (`LLAMA_MOE_CACHE_CPU_IDS_ON_CPU=1`), S10, branch `inkling-try-cpu-ids` | decode 36.4 vs controls 37.1 / 36.7 | neutral; the expert uploads, not the CPU part, are on the critical path |
| E3b: E3 plus CPU share 0.85 / 0.95 | decode 37.2 / 36.6 vs 36.8 / 37.1 | neutral |
| more CPU threads on S13 (`-t 8` / `-t 12`) | decode 36.5 / 35.5 vs 37.4 / 37.6 with `-t 6` | slower: the CPU expert part is RAM bound |
| E5: this token's expert uploads over a second stream so CPU part and PCIe run in parallel (`LLAMA_MOE_CACHE_FILL_STREAM=1`), branch `inkling-try-split-path` | decode 36.1 / 35.1 vs S13 37.5 | slower: the scheduler queues the CPU part's input copy behind the uploads; needs a scheduler change |
| thread polling on S13 (`--poll 100`, `--cpu-strict 1`) | decode 37.5 / 37.5 vs 37.2 / 37.2 | inside noise |
| `-t 5` / `--poll 100` on S16 | decode 37.8 / 38.2 vs 38.9 / 38.4 | slower / neutral |

| OpenMP wait policy on S13 (`OMP_WAIT_POLICY=ACTIVE`, `GOMP_SPINCOUNT=INFINITE`) | decode 37.7 / 37.3 vs 37.7 / 37.2 | neutral |
| E6: relative-position bias in the vec FA kernel for decode (`GGML_CUDA_FATTN_BIAS_VEC=1`), branch `inkling-try-fattn-vec-bias` | decode 37.2 / 37.4 vs 37.7 / 37.8 / 37.8 | -1 %: with GQA 3 the vec kernel decodes q4_0 K/V once per Q head (0.78 vs 0.69 ms/token) |
| E9: background-fill the CPU-computed misses of loop passes 1/2 for the later passes (`LLAMA_MOE_CACHE_LOOP_FILL=1`), branch `inkling-try-loop-fill` | S16 decode 37.9 / 37.6 vs 38.2 / 38.5 / 38.5 | -1.6 %: the fill is committed at the next layer, so the uploads land on the critical path |

Current status and ranked next levers: STATUS section of [PLAN.md](PLAN.md). Older plan: [NEXT.md](NEXT.md).

## Decode plateau (2026-11-22 08:44)

On the S10 preset, CPU share 0.6 / 0.7 / 0.8 and cache 3650 / 3900 all land at 34.3-35.2 t/s (noise).
Cache stats during decode (`-lv 4`): hit rate 83 %; of the misses ~2/3 run on the CPU, ~1/3 are uploaded
(~21 experts per token). Each of the 48 layer applications also syncs once to read the router ids, which leaves the
GPU idle while the host plans the cache. Removing that sync with PR #32076 (rebased, works now) made decode slower (32.1 vs 35.2), so the sync is not the
main cost either; the remaining time is in the misses (CPU experts and uploads).
