#!/bin/bash
# launch_emb2.sh — EmbeddingGemma 2 on Jetson 8GB (coexistence-tuned)
# Usage: bash launch_emb2.sh [port]   (default 8091)
# Retrieval: 8/8; ~1200 tok/s batches; 518MB RAM.
# MEAN POOLING IS MANDATORY (last pooling breaks discrimination 1/3 -> 3/3).
# Batch width = max doc length: b=512 fits typical RAG chunks (200-500 tok).
# For docs up to 2048 tok use b=2048 (RAM grows ~2.8GB at load; run alone).

PORT="${1:-8091}"

GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_NO_PINNED=1 \
setsid ~/llama.cpp/build/bin/llama-server \
    -m ~/models/embeddinggemma-2/embeddinggemma-2-UD-Q4_K_XL.gguf \
    --alias embeddinggemma-2 \
    --host 127.0.0.1 --port "$PORT" \
    -ngl 99 -c 2048 --embeddings --pooling mean \
    -b 512 -ub 512 -fa on -np 1 -t 6 --fit off \
    > /tmp/embeddinggemma-2-server.log 2>&1 &

echo "EmbeddingGemma 2 starting on :$PORT (log: /tmp/embeddinggemma-2-server.log)"
