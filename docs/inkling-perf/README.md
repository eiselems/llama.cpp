# Inkling-2-Mini perf log (fork branch `inkling-2-perf-improvement`)

RTX 3070 8 GB, Ryzen 5 5600X, 32 GB RAM. Unsloth `Inkling-2-Mini-UD-Q4_K_M`, 32k ctx, q8_0 KV, experts in RAM.
Every row is measured with `scripts/inkling-bench.sh` (one server session: greedy correctness check, then
~1k and ~4.8k token prompts with 128 generated tokens). Only rows that pass the correctness check are recorded.

![prefill and decode per step](progress.svg)

## Presets (2026-11-22 08:35)

VRAM peaks include the desktop. Check the idle GPU use first (`nvidia-smi`): it was ~1070 MiB on the evening of
2026-11-21 (heavy desktop) and ~560 MiB on the morning of 2026-11-22 (light desktop). The cache must shrink by about
the difference on a heavy desktop, or the server runs out of memory.

Common flags: `--fit off -ngl all -cmoe -b 2048 --moe-cache-cpu-share 0.7 -fa on -ctk q8_0 -ctv q8_0 -c 32768`

| preset | env + extra flags | -ub | cache MiB (light / heavy desktop) | prefill 1k / 4.8k | decode @1k / @4.8k | VRAM peak (light) |
|---|---|---:|---|---:|---:|---:|
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

Plan for the next session: [NEXT.md](NEXT.md).

## Decode plateau (2026-11-22 08:44)

On the S10 preset, CPU share 0.6 / 0.7 / 0.8 and cache 3650 / 3900 all land at 34.3-35.2 t/s (noise).
Cache stats during decode (`-lv 4`): hit rate 83 %; of the misses ~2/3 run on the CPU, ~1/3 are uploaded
(~21 experts per token). Each of the 48 layer applications also syncs once to read the router ids, which leaves the
GPU idle while the host plans the cache. Removing that sync with PR #32076 (rebased, works now) made decode slower (32.1 vs 35.2), so the sync is not the
main cost either; the remaining time is in the misses (CPU experts and uploads).
