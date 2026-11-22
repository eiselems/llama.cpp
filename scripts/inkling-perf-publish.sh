#!/bin/bash
# Regenerate the perf chart and commit + push ONLY docs/inkling-perf and the inkling bench scripts.
# Safe to run while other work is in progress: unrelated changes in the worktree are left untouched.
set -u
cd "$(dirname "$0")/.." || exit 1
# only publish from the main perf branch, never from a try branch (it would push untested commits)
branch=$(git symbolic-ref --short -q HEAD)
if [ "$branch" != "inkling-2-perf-improvement" ]; then
    echo "not publishing: on branch '${branch:-detached}', expected inkling-2-perf-improvement" >&2
    exit 1
fi
# copies of the hill-climb configs and ledger (experiment dir outside the repo)
E=${CLIMB_DIR:-$HOME/eval-runs-20261122-inkling/climb}
if [ -d "$E/configs" ]; then
    mkdir -p docs/inkling-perf/climb && cp "$E"/configs/*.json docs/inkling-perf/climb/ 2>/dev/null
    [ -s "$E/ledger.jsonl" ] && cp "$E/ledger.jsonl" docs/inkling-perf/climb/ledger.jsonl
fi
python3 -I scripts/inkling-perf-chart.py || exit 1
paths=(docs/inkling-perf scripts/inkling-bench.sh scripts/inkling-perf-chart.py scripts/inkling-perf-publish.sh scripts/inkling)
git add -- "${paths[@]}"
if git diff --cached --quiet -- "${paths[@]}"; then
    echo "nothing new to publish"
else
    msg="inkling perf : update progress log ($(tail -1 docs/inkling-perf/progress.tsv | cut -f1,3 | tr '\t' ' '))

$(cat docs/inkling-perf/progress-table.md)"
    git commit -q -m "$msg" -- "${paths[@]}" || exit 1
fi
git push -q fork HEAD:inkling-2-perf-improvement 2>&1 | tail -2
git log --oneline -1 | cat
