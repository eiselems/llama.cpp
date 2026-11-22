| step | change | commit | VRAM peak (MiB) | prefill 1k | prefill 4.8k | decode @1k | decode @4.8k |
|---|---|---|---:|---:|---:|---:|---:|
| S0 | upstream master (no-op-offload, ub32, cache 2800) | `afe77e18` | 6307 | 57.6 | 56.4 | 18 | 17.3 |
| S1 | + MoE cache dedup (adishik 65dd939f) | `2438cefc` | 6596 | 70.7 | 77.1 | 24.7 | 24.3 |
| S2 | + fused glue ops + FA bias (#32740), same flags | `4fb181db` | 6420 | 74.1 | 81.5 | 27.4 | 27.1 |
| S3 | op-offload on (views fix), ub512, cache 3400 | `4fb181db` | 7441 | 434.4 | 501.3 | 30.1 | 28.9 |
