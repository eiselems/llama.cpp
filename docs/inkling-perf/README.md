# Inkling-2-Mini perf log (fork branch `inkling-2-perf-improvement`)

RTX 3070 8 GB, Ryzen 5 5600X, 32 GB RAM. Unsloth `Inkling-2-Mini-UD-Q4_K_M`, 32k ctx, q8_0 KV, experts in RAM.
Every row is measured with `scripts/inkling-bench.sh` (one server session: greedy correctness check, then
~1k and ~4.8k token prompts with 128 generated tokens). Only rows that pass the correctness check are recorded.

![prefill and decode per step](progress.svg)

The same numbers as a table: [progress-table.md](progress-table.md) (raw: [progress.tsv](progress.tsv)).
