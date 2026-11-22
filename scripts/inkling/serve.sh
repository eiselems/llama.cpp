#!/bin/bash
# Start / stop an Inkling-2-Mini llama-server preset for scripts/inkling/bench.py.
#   serve.sh start PRESET [extra llama-server args...]   detached, PID in $RUN/server.pid, log in $RUN/server-PRESET.log
#   serve.sh stop                                         stop by PID: SIGTERM, wait up to 30 s, then SIGKILL
# Env: BIN (llama-server), MODEL, PORT (8089), RUN (state dir), CTX (32768).
# Presets (docs/inkling-perf/README.md): S10 decode first, S11 balanced, S12 max prefill, S13 = S10 + background fill. Model card sampling is
# set per request by bench.py (temperature 1.0, top-p 1.0, top-k 0, min-p 0).
set -u
here=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$here/build/bin/llama-server}
MODEL=${MODEL:-$HOME/sda6/models/inkling2/Inkling-2-Mini-UD-Q4_K_M.gguf}
PORT=${PORT:-8089}
RUN=${RUN:-$HOME/eval-runs-20261122-inkling/climb/run}
CTX=${CTX:-32768}
mkdir -p "$RUN"
pidf="$RUN/server.pid"

stop() {
    [ -s "$pidf" ] || return 0
    local pid; pid=$(cat "$pidf")
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
    fi
    rm -f "$pidf"
}

case "${1:-}" in
stop) stop ;;
start)
    preset=${2:?preset}; shift 2
    stop
    common=(--fit off -ngl all -cmoe -b 2048 --moe-cache-cpu-share 0.7 -fa on -ctk q4_0 -ctv q4_0)
    case "$preset" in
        S10) env=(LLAMA_EXPERT_PREFETCH=3); flags=(-ub 1024 --moe-cache-mib 3650) ;;
        S11) env=(LLAMA_EXPERT_PREFETCH=1); flags=(-ub 1024 --moe-cache-mib 3200) ;;
        S12) env=(LLAMA_EXPERT_PREFETCH=1); flags=(-ub 2048 --moe-cache-mib 2650) ;;
        S13) env=(LLAMA_EXPERT_PREFETCH=3 LLAMA_MOE_CACHE_ASYNC_FILL=1); flags=(-ub 1024 --moe-cache-mib 3650) ;;  # S10 + E4
        S14) env=(LLAMA_EXPERT_PREFETCH=1 LLAMA_EXPERT_PREFETCH_SKIP=1); flags=(-ub 1024 --moe-cache-mib 3200) ;;  # S11 + E2
        *)   echo "unknown preset $preset" >&2; exit 1 ;;
    esac
    # extra args after the preset override the preset (llama.cpp: the last flag wins); EXTRA_ENV adds env vars
    setsid env "${env[@]}" ${EXTRA_ENV:-} "$BIN" -m "$MODEL" -c "$CTX" -np 1 --load-mode none --jinja -t 6 \
        --host 127.0.0.1 --port "$PORT" --alias inkling2-mini "${common[@]}" "${flags[@]}" "$@" \
        > "$RUN/server-$preset.log" 2>&1 < /dev/null &
    echo $! > "$pidf"
    ;;
*) echo "usage: serve.sh start PRESET [args...] | stop" >&2; exit 1 ;;
esac
