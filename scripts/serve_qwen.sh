#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8104}"

# This intentionally uses the plain vLLM CLI. The live launcher imported the
# B300-specific b300_compat module; that private import is not portable.
exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.45 \
  --dtype bfloat16 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs \
  '{"enable_thinking": true, "reasoning_effort": "low"}'
