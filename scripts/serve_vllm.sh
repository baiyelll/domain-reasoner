#!/bin/bash
set -e

# ============================================================
# vLLM Multi-LoRA Serving
# Ready-to-run once you have a GPU
# ============================================================

# Path to your trained adapter (change if needed)
ADAPTER_PATH="${ADAPTER_PATH:-./adapters/swe-coding-agent-v1}"
BASE_MODEL="${BASE_MODEL:-unsloth/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"

echo "=== vLLM Multi-LoRA Server ==="
echo "Base model : $BASE_MODEL"
echo "Adapter    : $ADAPTER_PATH"
echo "Port       : $PORT"
echo ""

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "WARNING: Adapter path '$ADAPTER_PATH' not found."
    echo "Download it from Google Drive first or set ADAPTER_PATH."
fi

vllm serve "$BASE_MODEL" \
    --enable-lora \
    --lora-modules coding="$ADAPTER_PATH" \
    --max-lora-rank 16 \
    --dtype float16 \
    --max-model-len 2048 \
    --port "$PORT"
