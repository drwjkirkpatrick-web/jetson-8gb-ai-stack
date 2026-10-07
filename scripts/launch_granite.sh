#!/bin/bash
# launch_granite.sh — Granite 4.0-3B generator for the 8GB AI stack
# Usage: bash launch_granite.sh [port]   (default 8090)
# 32K ctx, q8_0 KV (MANDATORY for coexistence: f16 leaves 96MB, q8_0 leaves 1.38GB).
# 19.2 tok/s alone, 15-16 tok/s under concurrent embedding load.

PORT="${1:-8090}"

GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_NO_PINNED=1 \
setsid ~/llama.cpp/build/bin/llama-server \
    -m ~/models/new-zoo/granite4-3b.gguf \
    --alias granite4-3b \
    --host 127.0.0.1 --port "$PORT" \
    -ngl 99 -c 32768 -ctk q8_0 -ctv q8_0 \
    -b 512 -ub 512 -fa on --jinja -np 1 -t 6 \
    --temp 0.0 --top-k 1 --repeat-penalty 1.0 --fit off \
    > /tmp/granite4-3b-stack.log 2>&1 &

echo "Granite 4.0-3B starting on :$PORT (log: /tmp/granite4-3b-stack.log)"
