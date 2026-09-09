#!/usr/bin/env bash
# =============================================================================
#  bench.sh — reproduce the published numbers / 复现实测数字
#
#  Runs, in order:
#    1. single-stream decode  (thinking off / on)
#    2. concurrency sweep     C = 1, 2, 4, 8   (shared KV pool)
#    3. long-context needle   100K by default  (pass tokens as arguments)
#
#  Usage / 用法:
#    bash bench.sh                    # full suite, default 100K needle
#    bash bench.sh 50000 100000       # custom needle sizes
#    SKIP_NEEDLE=1 bash bench.sh      # skip the long-context test
#    CONCURRENCY="1 2 4" bash bench.sh
#
#  Output: human-readable log + a markdown report in results/
# =============================================================================
set -euo pipefail

PORT="${PORT:-8080}"
BASE="http://127.0.0.1:${PORT}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$HERE/scripts"
[ -d "$SCRIPTS" ] || SCRIPTS="$(dirname "$HERE")"   # installed layout fallback
RESULTS="${RESULTS:-$HERE/results}"
NEEDLES=("$@")
[ ${#NEEDLES[@]} -eq 0 ] && NEEDLES=(100000)
CONCURRENCY="${CONCURRENCY:-1 2 4 8}"

BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; OFF=$'\033[0m'
say() { echo; echo "${BOLD}==> $*${OFF}"; }

say "preflight"
curl -s -m 5 -o /dev/null "$BASE/health" || { echo "server not healthy at $BASE"; exit 1; }
MODEL=$(curl -s "$BASE/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
echo "    endpoint : $BASE"
echo "    model    : $MODEL"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/    gpu      : /'

mkdir -p "$RESULTS"
STAMP=$(date +%Y%m%d-%H%M%S)
REPORT="$RESULTS/bench-$STAMP.md"
LOG="$RESULTS/bench-$STAMP.log"

{
  echo "# Benchmark — $(date -Iseconds)"
  echo
  echo "- endpoint: \`$BASE\`"
  echo "- model: \`$MODEL\`"
  echo "- gpus:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/  - /'
  echo
} > "$REPORT"

# ------------------------------- 1. single stream ----------------------------
say "1/3 single-stream decode / 单流解码"
{
  echo "## 1. Single-stream decode / 单流解码"
  echo '```'
} >> "$REPORT"
python3 "$SCRIPTS/bench_gsq.py" 2>&1 | grep -vE '^needle' | tee -a "$LOG" | tee -a "$REPORT.tmp"
grep -vE '^needle' "$REPORT.tmp" >> "$REPORT" 2>/dev/null || true
{ echo '```'; echo; } >> "$REPORT"
rm -f "$REPORT.tmp"

# ------------------------------- 2. concurrency ------------------------------
say "2/3 concurrency sweep / 并发扫描"
{
  echo "## 2. Concurrency (shared paged KV pool) / 并发（共享 KV 池）"
  echo
  echo '```'
} >> "$REPORT"
python3 "$SCRIPTS/bench_concurrency.py" 2>&1 | tee -a "$LOG" | tee -a "$REPORT.tmp"
grep -vE '^model:' "$REPORT.tmp" >> "$REPORT" 2>/dev/null || true
{ echo '```'; echo; } >> "$REPORT"
rm -f "$REPORT.tmp"

# ------------------------------- 3. long context -----------------------------
if [ "${SKIP_NEEDLE:-0}" = "1" ]; then
  say "3/3 long-context needle — skipped (SKIP_NEEDLE=1)"
else
  say "3/3 long-context needle / 长文召回"
  {
    echo "## 3. Long-context needle / 长文召回"
    echo
    echo '```'
  } >> "$REPORT"
  python3 "$SCRIPTS/bench_gsq.py" "${NEEDLES[@]}" 2>&1 | grep -E '^needle' | tee -a "$LOG" | tee -a "$REPORT.tmp"
  cat "$REPORT.tmp" >> "$REPORT" 2>/dev/null || true
  { echo '```'; echo; } >> "$REPORT"
  rm -f "$REPORT.tmp"
fi

echo
echo "${GREEN}report:${OFF} $REPORT"
echo "${GREEN}raw log:${OFF} $LOG"
