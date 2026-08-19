#!/usr/bin/env bash
# Jinpeng Zhai 914962409@qq.com
#
# Manage the llama.cpp judge servers for the SKAG judge-validation runs.
# (LM Studio stopped shipping SM70 runtimes, so the V100 judges now run on
# llama.cpp's OpenAI-compatible llama-server instead.)
#
# Usage:
#   ./rag/judge_server.sh start  <qwen36|gemma4|both>   load a judge server
#   ./rag/judge_server.sh stop   <qwen36|gemma4|all>
#   ./rag/judge_server.sh status
#
# Rig layout (do not change without reason):
#   - GPU 0/1: free NVLink pair -> the judges (tensor-split across both)
#   - GPU 2/3: vLLM (TP=2, :8010) -> excluded via CUDA_VISIBLE_DEVICES
#   - --reasoning off: the original judge runs were thinking-suppressed;
#     with llama.cpp this is a SERVER flag (no per-request knob)
#   - --parallel 2: llama-server DIVIDES --ctx-size across slots, so the
#     total 8192 tokens of KV become 2 x 4096/slot (was 4 x 2048). The
#     2048/slot setting truncated long judge prompts (~1965-2225 tok) at
#     the 2048 boundary: output got finish_reason=length with an unclosed
#     JSON -> permanently unparseable at seed 0. With 4096/slot every
#     prompt in the row has ~1870 tok headroom. Total KV (and VRAM) is
#     unchanged; a 3rd judge.py worker just queues briefly.
#
# Both judges fit resident at once (Qwen ~16.4G + Gemma ~9.6G per GPU),
# so stability runs (Q1-Q5) and Gemma runs (G0-G2) need no model swapping.
set -euo pipefail

LLAMA_CPP="$HOME/llama.cpp"
SERVER="$LLAMA_CPP/build/bin/llama-server"
MODELS="$HOME/models/HauhauCS"
GPUS="0,1"
PORT_QWEN=1234
PORT_GEMMA=1235
CTX=8192

qwen_gguf="$MODELS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced/Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q8_K_P.gguf"
gemma_gguf="$MODELS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"

log_file() { echo "/tmp/judge_server_$1.log"; }

up() {  # $1 = port
  curl -sf -m 2 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1
}

start_judge() {  # $1 = name (qwen36|gemma4)
  local name=$1 gguf port
  case $name in
    qwen36)  gguf=$qwen_gguf;  port=$PORT_QWEN ;;
    gemma4)  gguf=$gemma_gguf;  port=$PORT_GEMMA ;;
  esac

  if up $port; then
    echo "$name already running on :$port"
    return 0
  fi

  echo "starting $name on :$port ($GPUS) ..."
  CUDA_VISIBLE_DEVICES=$GPUS setsid nohup "$SERVER" \
    -m "$gguf" \
    --split-mode tensor \
    -ngl -1 \
    --ctx-size $CTX \
    --port $port \
    --alias "$name" \
    --reasoning off \
    --parallel 2 \
    --cont-batching \
    >"$(log_file $name)" 2>&1 &

  local i
  for i in $(seq 1 120); do
    if up $port; then
      echo "$name ready on :$port (log: $(log_file $name))"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: $name did not come up within 240s; see $(log_file $name)" >&2
  return 1
}

stop_judge() {  # $1 = name
  local name=$1 port
  case $name in
    qwen36) port=$PORT_QWEN ;;
    gemma4) port=$PORT_GEMMA ;;
  esac
  local pids
  pids=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | sort -u || true)
  if [ -n "$pids" ]; then
    echo "stopping $name (pids: $pids)"
    kill $pids
    for i in $(seq 1 30); do
      up $port || return 0
      sleep 1
    done
    echo "ERROR: $name still up on :$port; kill -9 $pids" >&2
    return 1
  fi
  echo "$name not running"
}

case ${1:-} in
  start)
    case ${2:-both} in
      qwen36|gemma4) start_judge $2 ;;
      both)         start_judge qwen36; start_judge gemma4 ;;
      *) echo "usage: $0 start <qwen36|gemma4|both>" >&2; exit 1 ;;
    esac
    ;;
  stop)
    case ${2:-all} in
      qwen36|gemma4) stop_judge $2 ;;
      all)          stop_judge qwen36; stop_judge gemma4 ;;
      *) echo "usage: $0 stop <qwen36|gemma4|all>" >&2; exit 1 ;;
    esac
    ;;
  status)
    for name in qwen36 gemma4; do
      case $name in
        qwen36) port=$PORT_QWEN ;;
        gemma4) port=$PORT_GEMMA ;;
      esac
      if up $port; then
        info=$(curl -sf -m 2 "http://127.0.0.1:$port/v1/models" | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"][0]; print(d["meta"]["size"]//2**30, "GiB,", d["meta"]["quantization_level"])' 2>/dev/null || echo '?')
        echo "$name: UP on :$port ($info)"
      else
        echo "$name: down"
      fi
    done
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    ;;
  *)
    echo "usage: $0 {start <qwen36|gemma4|both>|stop <qwen36|gemma4|all>|status}" >&2
    exit 1
    ;;
esac
