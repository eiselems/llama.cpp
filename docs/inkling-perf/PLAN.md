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
