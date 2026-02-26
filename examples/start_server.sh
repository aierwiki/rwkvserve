#!/bin/bash
#
# RWKVServe - Start Inference Server
#
# Starts an OpenAI-compatible API server for RWKV models.
# Supports base model + LoRA adapter (vLLM-compatible CLI).
#
# Usage:
#   chmod +x start_server.sh
#   ./start_server.sh
#

set -e

# ==================== Configuration ====================

# Base model path
BASE_MODEL="output_sft_grpo/rwkv7-epoch3"

# Base model name in API
MODEL_NAME="rwkv-7"

# LoRA adapter (set ENABLE_LORA=1 to use)
ENABLE_LORA=1
LORA_NAME="rwkv-7-grpo"
LORA_PATH="output_grpo/rwkv7-batch1728"

# Server settings
HOST="0.0.0.0"
PORT=8000

# Inference settings
MAX_BATCH_SIZE=32
PREFILL_CHUNK_SIZE=512
DTYPE="bf16"

# State Cache settings
CACHE_LEVEL="prefix"
MAX_CACHE_MEMORY_GB=4.0

# GPU device (hardcoded)
export CUDA_VISIBLE_DEVICES=6

# ==================== No modification needed below ====================

echo "=================================================="
echo "       RWKVServe - Starting Inference Server"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Base model:         ${BASE_MODEL}"
echo "  Model name:         ${MODEL_NAME}"
if [ "${ENABLE_LORA}" = "1" ]; then
    echo "  LoRA adapter:       ${LORA_NAME} (${LORA_PATH})"
fi
echo "  Server:             http://${HOST}:${PORT}"
echo "  Max batch size:     ${MAX_BATCH_SIZE}"
echo "  Prefill chunk:      ${PREFILL_CHUNK_SIZE}"
echo "  Dtype:              ${DTYPE}"
echo "  Cache level:        ${CACHE_LEVEL}"
echo "  Cache memory:       ${MAX_CACHE_MEMORY_GB} GB"
echo "  GPU device:         ${CUDA_VISIBLE_DEVICES}"
echo ""

# Check base model path
if [ ! -e "${BASE_MODEL}" ]; then
    echo "Error: Base model path does not exist: ${BASE_MODEL}"
    exit 1
fi

# Check LoRA path
if [ "${ENABLE_LORA}" = "1" ] && [ ! -e "${LORA_PATH}" ]; then
    echo "Error: LoRA adapter path does not exist: ${LORA_PATH}"
    exit 1
fi

echo "Starting server..."
echo ""
echo "API endpoints:"
echo "  GET  http://${HOST}:${PORT}/v1/models"
echo "  POST http://${HOST}:${PORT}/v1/chat/completions"
echo "  POST http://${HOST}:${PORT}/v1/completions"
echo "  GET  http://${HOST}:${PORT}/health"
echo "  GET  http://${HOST}:${PORT}/docs   (Swagger UI)"
echo ""
if [ "${ENABLE_LORA}" = "1" ]; then
    echo "Use model='${LORA_NAME}' in API requests to select the LoRA adapter"
    echo ""
fi
echo "Press Ctrl+C to stop the server"
echo ""

# Build command
CMD="rwkvserve"
CMD+=" --model-path ${BASE_MODEL}"
CMD+=" --model-name ${MODEL_NAME}"
CMD+=" --host ${HOST}"
CMD+=" --port ${PORT}"
CMD+=" --dtype ${DTYPE}"
CMD+=" --max-batch-size ${MAX_BATCH_SIZE}"
CMD+=" --prefill-chunk-size ${PREFILL_CHUNK_SIZE}"
CMD+=" --cache-level ${CACHE_LEVEL}"
CMD+=" --max-cache-memory ${MAX_CACHE_MEMORY_GB}"

if [ "${ENABLE_LORA}" = "1" ]; then
    CMD+=" --enable-lora"
    CMD+=" --lora-modules ${LORA_NAME}=${LORA_PATH}"
fi

# Enable reasoning content extraction (<think>...</think> -> reasoning_content)
CMD+=" --enable-reasoning --reasoning-parser deepseek_r1"

eval ${CMD}
