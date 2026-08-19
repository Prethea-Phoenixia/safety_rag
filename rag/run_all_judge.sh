#!/usr/bin/env bash
# Jinpeng Zhai 914962409@qq.com
#
# Run the judge-validation matrix (2026-08-19 revised plan):
#
#   Q1-Q3  Qwen 3.6   200-sample subset, baseline + ours only   (3 x 400)
#   G0     Gemma-4    FULL row, baseline + ours, ONCE           (1 x 1864)
#
# (Original plan was Q1-Q5/G1-G2 all-3-types + Gemma full all-3-types;
# cut to baseline + ours to cut wall clock. The all-3-types headline
# numbers already exist as Q0 in the JUDGED_*_internvl3_5-8b.json files.)
#
# Both judges run IN PARALLEL: they are loaded on separate ports (:1234
# qwen36, :1235 gemma4) sharing the GPU 0/1 tensor split. The Q branch and
# the G branch run as two concurrent background processes; their logs go to
# /tmp/run_all_judge_q.log and /tmp/run_all_judge_g.log.
#
# Misformed-JSON handling: the judge is stochastic and occasionally emits
# unparseable output -> judge.py's resume skips only fully-judged samples, so
# run_one re-fires the same command up to $MAX_ROUNDS times until
# check_pending.py reports 0.
#
# Usage:
#   ./rag/run_all_judge.sh              # both branches, in parallel
#   ./rag/run_all_judge.sh Q            # Qwen subset stability runs only
#   ./rag/run_all_judge.sh G            # Gemma full-row run only
#
# Both judge servers must be up first:  ./rag/judge_server.sh status
# Long-running: launch as a Hermes background process.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/miniconda3/envs/safety_rag/bin/python"
SUBSET="subset_200.json"
WORKERS=3
MAX_ROUNDS=${MAX_ROUNDS:-5}
VLM="internvl3_5-8b"
TYPES="baseline,ours"
# dataset display names (judge.py matches display_name OR prefix)
DATASETS=(SIUO "MSS-Bench" "MM-Safety" "SPA-VL_Harm")
Q_RUNS=(1 2 3)

# pending <judge-out-suffix> <subset-or-empty> <dataset>
pending() {
  local out=$1 subset=$2 dataset=$3
  (cd "$HERE" && "$PY" check_pending.py --dataset "$dataset" --vlm "$VLM" \
    --out "$out" --types "$TYPES" ${subset:+--subset "$subset"})
}

# run_one <judge> <out-suffix> <subset-or-empty> <dataset>
run_one() {
  local judge=$1 out=$2 subset=$3 dataset=$4
  local args=(--dataset "$dataset" --vlm "$VLM" --judge "$judge" --out "$out" \
              --types "$TYPES" --num_workers "$WORKERS")
  [ -n "$subset" ] && args+=(--subset "$subset")
  local round p
  for round in $(seq 1 "$MAX_ROUNDS"); do
    (cd "$HERE" && "$PY" judge.py "${args[@]}") </dev/null >/dev/null 2>&1
    p=$(pending "$out" "$subset" "$dataset") || p="?1"
    if [ "$p" = "0" ]; then
      echo "[done] $judge $out $dataset"
      return 0
    fi
    if [ "$round" -lt "$MAX_ROUNDS" ]; then
      echo "[retry $((round + 1))/$MAX_ROUNDS] $judge $out $dataset: $p pending"
    fi
  done
  echo "[WARN] $judge $out $dataset: $p calls still pending after $MAX_ROUNDS rounds"
  return 1
}

# run_qwen: Q1-Q3 subset stability runs; run_gemma: G0 full row, once.
# Datasets are done sequentially per branch (one model per port); the two
# branches themselves run in parallel (see the ALL case below).
run_qwen() {
  local r d
  for r in "${Q_RUNS[@]}"; do
    for d in "${DATASETS[@]}"; do
      run_one qwen36 "_qwen36_run$r" "$SUBSET" "$d"
    done
  done
}

run_gemma() {
  local d
  for d in "${DATASETS[@]}"; do
    run_one gemma4 "_gemma4_full" "" "$d"
  done
}

mode=${1:-ALL}
declare -a pids=() names=()

case $mode in
  Q)
    run_qwen
    ;;
  G)
    run_gemma
    ;;
  ALL)
    (run_qwen)  >/tmp/run_all_judge_q.log 2>&1 & pids+=($!); names+=("Q")
    (run_gemma) >/tmp/run_all_judge_g.log 2>&1 & pids+=($!); names+=("G")
    rc=0
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        echo "[branch ${names[$i]}] finished OK (log: /tmp/run_all_judge_${names[$i],,}.log)"
      else
        echo "[branch ${names[$i]}] had pending leftovers (log: /tmp/run_all_judge_${names[$i],,}.log)"
        rc=1
      fi
    done
    ;;
  *) echo "usage: $0 [Q|G|ALL]" >&2; exit 1 ;;
esac

# Final tally
echo "=== final pending tally (types: $TYPES) ==="
for entry in "_qwen36_run1:$SUBSET" "_qwen36_run2:$SUBSET" "_qwen36_run3:$SUBSET" "_gemma4_full:"; do
  out_suffix=${entry%%:*}; subset=${entry#*:}
  for d in "${DATASETS[@]}"; do
    p=$(pending "$out_suffix" "$subset" "$d")
    echo "$out_suffix $d: ${p} pending"
  done
done
echo "=== run complete ==="
exit ${rc:-0}
