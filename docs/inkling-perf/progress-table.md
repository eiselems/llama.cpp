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
| Q9 | IQ4_NL quant (S19 = S18 on IQ4_NL) | `db736005` | 7424 | 873 | 1073 | 35.7 | 26.6 |
| T10 | S19 cache 2650 -> 2900 (IQ4_NL frees VRAM) | `4b939d81` | 7670 | 890 | 1080 | 36.7 | 28.7 |
| S19c | S19 as before 2026-11-23 (auto cache, half-life 128) | `edc5907e` | 7782 | 865.5 | 1058.9 | 35.09 | 26.09 |
| S24 | + half-life 64, mask from positions, K/V chunks (32k) | `edc5907e` | 7772 | 870.9 | 1069.7 | 36.05 | 27.42 |
| S25 | + host experts on huge pages (THP) | `c0fe97986` | 7782 | 893.8 | 1089.8 | 37.85 | 28.6 |
| S26 | + async CPU result upload, prefetched expert ids (exact) | `41a7235a2` | 7772 | 895.1 | 1086.4 | 38.79 | 29.26 |
| S27 | + single staging slot (prefetch 3): +3.0 % tg vs S26 same session | `aa5671ce2` | 7782 | 808.7 | 1035.7 | 38.48 | 29.55 |
| S28 | + K1 attention: bias table view + sliding-window block skipping (exact) | `bf11e34d7` | 7782 | 835.7 | 1193.6 | 40.14 | 28.55 |
| S29 | + K2 decode attention straight from q4_0 K/V | `2be7815c5` | 7782 | 829.6 | 1194.4 | 41.00 | 38.57 |
| S30 | + K3 sinks subtracted in the attention kernels (exact) | `b2d1c04ba` | 7782 | 829.9 | 1220.1 | 41.40 | 38.0 |
| S31 | + K4 fused expert SiTU in prefill (exact) | `b1805749b` | 7782 | 832.8 | 1288.1 | 41.42 | 39.15 |
| S32 | + K6 q8_1 input reuse in decode (exact) | `bf030eb61` | 7782 | 843.6 | 1291.3 | 41.81 | 39.68 |
| S36 | + sampling on the GPU (--backend-sampling) | `0de686ca7` | 7823 | 820.3 | 1267.0 | 42.58 |  |
| S37 | + K10 far K/V blocks of the position bias in prefill attention (exact) | `aa67a4377` | 7823 | 838.2 | 1296.0 | 44.28 | 40.11 |
