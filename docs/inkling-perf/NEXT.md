# Next steps (as of 2026-11-22 04:30)

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

**Status 04:39: works and merged as opt-in LLAMA_EXPERT_PREFETCH=1** (bug was the NVFP4 `.scale` tensors in the
recorded order). +16 % prefill at -ub 1024. Done 04:41: -ub 2048 = 925 t/s (S7). Tried 04:50: prefetching only gate/up (1 slot) is WORSE than no prefetch, so smaller slots must split both tensors into expert ranges instead. Next: try it at -ub 2048 (expected bigger gain, compute share is
larger), shrink the slots (prefetch per expert range instead of whole tensors), and check the hit rate stays ~100 %.

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

Prototype tried 2026-11-22 04:26 on branch `inkling-prefetch-try` (pushed): llama-level prefetch in the copy
callback (second CUDA backend instance = second stream, 2 staging slots, device-to-device copy on use).
Result: no gain at 1k (561.7 vs ~575 t/s) and an assert on the 4.8k prompt (slot sizing must use
ggml_backend_buft_get_alloc_size, CUDA pads quantized tensors). Before fixing it, find out why there is no overlap:
likely the MoE cache's per-layer plan() sync still serializes every split. Try the prototype with the MoE cache off
first to separate the two effects.

Follow-up 04:30 (slot sizing fixed on the try branch): with the MoE cache OFF prefetch is still no faster
(466.9 / 582.1 vs 464.3 / 578.7 t/s), while the slots are allocated (+790 MiB), so the cache sync is not the cause.
**Lead:** the host experts are probably NOT pinned (22 GB/s alone does not prove it). cudaMemcpyAsync from pageable
memory blocks the host thread, so nothing overlaps. First check which buffer type holds the host experts
(`-lv 4` load log: CPU vs CUDA_Host); pinning them may also speed up the plain upload path.
**Update 04:34:** pinning ruled out - the host experts are in a `CUDA_Host` (pinned) buffer of 14563 MiB. The real
cause: the prefetch hit rate is 6 % (62 of 960 expert tensors from staging), so the prototype almost never runs ahead.
The next-tensor prediction (order of the previous graph, matched by position) does not line up; debug that first by
logging the src names per position for two consecutive ubatches. The overlap idea itself is still untested.
Side finding: the MoE cache also speeds up prefill (464 t/s without vs ~575 with a 2600-3000 MiB cache at -ub 1024).

## 2. Make --fit pick the good preset

Default --fit gives 365 t/s / 20 t/s because it picks a small MoE cache. It should prefer
`-ub 1024 --moe-cache-mib ~3400 --moe-cache-cpu-share 0.7` on this card (see README sweep).

## 3. DSpark speculative decoding (PR #32218 + #32773)

Measured: slower than without (31.3 vs 34.1 t/s at best). On branch `inkling-dspark-try` (local only). Needs ~0.5 GiB VRAM (draft model + q4_0 draft KV +
a 72 MiB layer-input buffer), so the MoE cache has to shrink. See the README for the measured result.

## 4. Smaller items

- `--moe-cache-cpu-share` swept: 0.6 = 33.6, 0.7 = 34.1, 0.8 = 33.2, 1.0 = 22 t/s; 0.7 kept.
- PR #32076 (MoE cache without syncs) crashes with dedup (illegal memory access) and gives no gain without it
  on Linux CUDA; parked on local branch `inkling-perf-32076-wip`.
- Card-sampling quality eval (v1-v4) on this build: speed no longer blocks it.
