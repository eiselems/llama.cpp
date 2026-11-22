# Next steps (as of 2026-11-22 ~03:20)

## Where the time goes now

Prefill with op offload uploads every expert of every layer application once per ubatch:
13.8 GiB (all experts) + 2 x 6.4 GiB (the 12 looped layers again in passes 2 and 3) = ~26.6 GiB.
At the ~22 GB/s this PCIe 4.0 x16 link reaches (the host experts are already pinned) that is ~1.3 s per ubatch.

| -ub | measured time per ubatch | upload | compute (rest) | upload share |
|---:|---:|---:|---:|---:|
| 512 | ~1.1 s | ~1.3 s minus cache hits | small | ~all |
| 1024 | ~1.75 s | ~1.3 s | ~0.45 s | ~75 % |
| 2048 | ~2.6 s | ~1.3 s | ~1.3 s | ~50 % |

Today each expert split runs strictly in sequence on one CUDA stream:
wait for previous split -> synchronize to read the router ids -> upload (GPU idle) -> compute (PCIe idle).

## 1. Overlap the expert upload with compute (largest prefill lever)

Expected: up to ~+34 % at -ub 1024, up to ~2x (~1500 t/s on 4.8k prompts) at -ub 2048.

Design sketch (ggml-backend scheduler + llama_context copy callback):
- For large batches (op offload, n_tokens >= 512) upload ALL experts of the next host-weight split without waiting
  for its router ids (at -ub >= 512 almost every expert is selected anyway). This removes the dependency on the
  previous split's result, so the upload can start before it.
- Upload on a second CUDA stream: a second ggml backend instance on the same device has its own stream and can
  write into buffers of the same buffer type.
- The copies must not live in gallocr memory (gallocr reuses regions across splits, an early upload would overwrite
  live tensors). Use a 2-slot staging ring sized for the largest host weight (~0.53 GiB gate_up of layers 20-23),
  outside the compute buffer; the compute buffer shrinks by about the same amount, so the net VRAM cost is ~1 slot.
- Events: copy stream waits on "split that used this slot finished" (recorded on the compute stream), compute
  stream waits on "upload into this slot finished" (recorded on the copy stream).
- Validate with scripts/inkling-bench.sh (greedy correctness) plus a 4.8k-prompt greedy diff against the current build.

## 2. Make --fit pick the good preset

Default --fit gives 365 t/s / 20 t/s because it picks a small MoE cache. It should prefer
`-ub 1024 --moe-cache-mib ~3400 --moe-cache-cpu-share 0.7` on this card (see README sweep).

## 3. DSpark speculative decoding (PR #32218 + #32773)

Measured: slower than without (31.3 vs 34.1 t/s at best). On branch `inkling-dspark-try` (local only). Needs ~0.5 GiB VRAM (draft model + q4_0 draft KV +
a 72 MiB layer-input buffer), so the MoE cache has to shrink. See the README for the measured result.

## 4. Smaller items

- `--moe-cache-cpu-share` between 0.7 and 1.0 was not swept finely (0.7 = 34.1 t/s, 1.0 = 22 t/s).
- PR #32076 (MoE cache without syncs) crashes with dedup (illegal memory access) and gives no gain without it
  on Linux CUDA; parked on local branch `inkling-perf-32076-wip`.
- Card-sampling quality eval (v1-v4) on this build: speed no longer blocks it.
