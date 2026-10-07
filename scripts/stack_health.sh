#!/bin/bash
# stack_health.sh — one-shot status of the full 8GB AI stack
echo "=== Jetson 8GB AI Stack health ==="
free -h | head -2
echo ""
for spec in "8090:Granite-4.0-3B" "8091:EmbeddingGemma-2"; do
    port="${spec%%:*}"; name="${spec#*:}"
    if curl -s -m 5 "http://127.0.0.1:${port}/health" | grep -q ok; then
        echo "  [OK]  :${port}  ${name}"
    else
        echo "  [DOWN] :${port}  ${name}"
    fi
done
ps aux | grep -i "[s]erena" | grep -qv grep && echo "  [OK]  Serena MCP running" || echo "  [DOWN] Serena MCP not detected"
echo ""
echo "MemAvailable: $(awk '/MemAvailable/ {print $2/1024 " MB"}' /proc/meminfo)"
