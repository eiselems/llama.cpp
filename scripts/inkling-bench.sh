#!/bin/bash
# Inkling-2-Mini benchmark + correctness check for the inkling-2-perf-improvement branch.
#
# usage: scripts/inkling-bench.sh LABEL [server args...]
#   env: MODEL (default: Unsloth UD-Q4_K_M), BIN (default: build/bin/llama-server), PORT (8089), OUT (results dir)
#
# One server session, run in this order:
#   1. correctness: greedy answers to "2+3" and a 1k-token summary prompt, compared by keyword
#   2. prefill 1k: ~1k-token prompt, 128 generated tokens (ignore_eos)  -> pp1k, tg
#   3. prefill 4.8k: ~4.8k-token prompt, 128 generated tokens           -> pp5k, tg@5k
# Peak VRAM is sampled every 200 ms over the whole session. One line per run goes to $OUT/inkling-bench.tsv.
set -u
label=$1; shift
here=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-$HOME/sda6/models/inkling2/Inkling-2-Mini-UD-Q4_K_M.gguf}
BIN=${BIN:-$here/build/bin/llama-server}
PORT=${PORT:-8089}
OUT=${OUT:-$HOME/eval-runs-20261122-inkling/bench}
mkdir -p "$OUT"
log="$OUT/$label.log"
args=("$@")
[ ${#args[@]} -eq 0 ] && args=(--fit off -ngl all -cmoe -b 2048 -ub 512 --moe-cache-mib 3400)

"$BIN" -m "$MODEL" -c 32768 -np 1 --load-mode none --jinja -t 6 -fa on -ctk q8_0 -ctv q8_0 \
    --host 127.0.0.1 --port "$PORT" "${args[@]}" > "$log" 2>&1 &
srv=$!
( peak=0; while kill -0 $srv 2>/dev/null; do
    v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$v" ] && [ "$v" -gt "$peak" ] && { peak=$v; echo "$peak" > "$OUT/.peak-$label"; }
    sleep 0.2; done ) &
mon=$!
stop() { kill $srv 2>/dev/null; sleep 3; kill -9 $srv 2>/dev/null; wait $srv 2>/dev/null; kill $mon 2>/dev/null; wait $mon 2>/dev/null; }

for _ in $(seq 300); do curl -sf -m2 "http://127.0.0.1:$PORT/health" >/dev/null && break; kill -0 $srv 2>/dev/null || break; sleep 1; done
if ! curl -sf -m2 "http://127.0.0.1:$PORT/health" >/dev/null; then
    err=$(grep -m1 -E ' E |GGML_ASSERT|CUDA error' "$log" | cut -c1-160)
    stop; printf '%s\t%s\tFAILED_TO_START\t%s\n' "$(date -Is)" "$label" "$err" | tee -a "$OUT/inkling-bench.tsv"; exit 1
fi

req() { # prompt max_tokens greedy(0/1)
    jq -n --arg c "$1" --argjson n "$2" --argjson g "$3" \
        '{messages:[{role:"user",content:$c}],max_tokens:$n,cache_prompt:false,chat_template_kwargs:{reasoning_effort:"none"}}
         + (if $g == 1 then {temperature:0} else {ignore_eos:true,temperature:1.0,top_p:1.0,top_k:0,min_p:0} end)' \
    | curl -s -m1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @-
}
text() { jq -r '((.choices[0].message.reasoning_content // "") + " " + (.choices[0].message.content // "")) | ascii_downcase'; }

fox1k=$(python3 -c "print('The quick brown fox jumps over the lazy dog near the old river bank. '*58 + '\nSummarize the text above in one sentence.')")
fox5k=$(python3 -c "print('The quick brown fox jumps over the lazy dog near the old river bank. '*280 + '\nSummarize the text above in one sentence.')")

# 1. correctness (greedy)
a=$(req "What is 2+3? Answer in one sentence." 160 1 | text)
b=$(req "$fox1k" 160 1 | text)
ok=PASS
grep -qE '(^|[^0-9])5([^0-9]|$)' <<< "$a" || ok="FAIL(2+3)"
{ grep -q 'fox' <<< "$b" && grep -qE 'repeat|summar' <<< "$b"; } || ok="$ok FAIL(summary)"
# a looping or degenerate answer: the same word 8+ times in a row
grep -qE '\b(\w+)( \1){7,}\b' <<< "$a $b" && ok="$ok FAIL(loop)"

# 2./3. speed (card sampling, fixed length)
r1=$(req "$fox1k" 128 0); r5=$(req "$fox5k" 128 0)
pp1=$(jq -r '.timings.prompt_per_second|.*10|floor/10' <<< "$r1"); tg1=$(jq -r '.timings.predicted_per_second|.*10|floor/10' <<< "$r1")
pp5=$(jq -r '.timings.prompt_per_second|.*10|floor/10' <<< "$r5"); tg5=$(jq -r '.timings.predicted_per_second|.*10|floor/10' <<< "$r5")
n5=$(jq -r '.timings.prompt_n' <<< "$r5")
stop
peak=$(cat "$OUT/.peak-$label" 2>/dev/null || echo 0); rm -f "$OUT/.peak-$label"
commit=${COMMIT:-$(git -C "$here" log --oneline -1 --format=%h)}

printf '%s\t%s\t%s\tVRAM_peak=%sMiB\tpp1k=%s\ttg1k=%s\tpp%s=%s\ttg@5k=%s\t%s\targs=%s\n' \
    "$(date -Is)" "$label" "$commit" "$peak" "$pp1" "$tg1" "$n5" "$pp5" "$tg5" "$ok" "${args[*]}" | tee -a "$OUT/inkling-bench.tsv"
