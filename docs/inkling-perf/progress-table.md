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
