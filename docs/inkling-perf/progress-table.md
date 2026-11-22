| step | change | commit | VRAM peak (MiB) | prefill 1k | prefill 4.8k | decode @1k | decode @4.8k |
|---|---|---|---:|---:|---:|---:|---:|
| S0 | upstream master (no-op-offload, ub32, cache 2800) | `afe77e18` | 6307 | 57.6 | 56.4 | 18 | 17.3 |
| S1 | + MoE cache dedup (adishik 65dd939f) | `2438cefc` | 6596 | 70.7 | 77.1 | 24.7 | 24.3 |
| S2 | + fused glue ops + FA bias (#32740), same flags | `4fb181db` | 6420 | 74.1 | 81.5 | 27.4 | 27.1 |
| S3 | op-offload on (views fix), ub512, cache 3400 | `4fb181db` | 7441 | 434.4 | 501.3 | 30.1 | 28.9 |
| S4 | + dedup with CPU share 0.7 | `2ecddfadf` | 7534 | 453.8 | 502.5 | 34.1 | 32.8 |
| S5 | preset: ub 1024 (balanced) | `1410a556` | 7758 | 582.4 | 685.8 | 34.1 | 32.1 |
| S6 | + expert prefetch (opt-in), cache 2600 | `d2514c6ea` | 7753 | 672.9 | 798.5 | 31.4 | 29.8 |
| S7 | prefetch + ub 2048, cache 2000 (max prefill) | `d25dc474` | 7620 | 659.8 | 924.9 | 29.2 | 26.7 |
| S8 | single-slot prefetch, cache 3400 (new balanced) | `dee6c0f51` | 7692 | 632.4 | 723.9 | 33.9 | 31.8 |
| S9 | prefetch ub 2048, cache 2400 (max prefill, light desktop) | `aff1063b1` | 7504 | 664.5 | 934.5 | 31.1 | 28.5 |
| S10 | balanced + q4_0 KV, cache 3650 | `dafda2ad7` | 7604 | 637.5 | 723.9 | 35.2 | 33 |
| S11 | 2-slot prefetch + q4_0 KV, cache 3200 | `355ff0d1d` | 7502 | 682.2 | 802.2 | 33.4 | 31.6 |
| S12 | max prefill + q4_0 KV, ub 2048, cache 2650 | `6b6115871` | 7382 | 676.2 | 932 | 32 | 27.5 |
| B10 | bench.py baseline S10 (median of 6, 3 runs) | `bench.py` | 7526 | 688 | 735 | 36.9 | 27.4 |
| B11 | bench.py baseline S11 | `bench.py` | 7424 | 745 | 820 | 35.3 | 26.9 |
| B12 | bench.py baseline S12 | `bench.py` | 7301 | 737 | 950 | 33.1 | 27.4 |
| E4 | + background cache fill (S13 = S10 + E4) | `d0c94745` | 7526 | 689 | 734 | 37.8 | 29 |
| E2 | + prefetch skips cached experts (S14 = S11 + E2) | `1bf03dedd` | 7424 | 858 | 912 | 35.9 | 28.1 |
| S15 | max prefill + prefetch skip (S15 = S12 + E2) | `c6c810f7d` | 7301 | 850 | 1003 | 33.2 | 25 |
| S16 | max decode: S13 at -ub 512, cache 3950 (prefill halves) | `1f01e1aaa` | 7660 | 367 | 379 | 38.4 | 31 |
| E7 | + MMA attention bias (S17 = S15 + E7) | `a39686409` | 7301 | 845 | 996 | 33.6 | 26.3 |
| E8 | + bias index from positions (S18 = S17 + E8) | `58797ab4f` | 7022 | 861 | 1057 | 33.4 | 24.9 |
