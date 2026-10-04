#!/bin/bash
# Dedicated NVIDIA RTX 4060 Context Synthesis & Compaction Server (Port 9001)
export CUDA_VISIBLE_DEVICES=0
exec /home/safiyu/llama.cpp-cuda/build/bin/llama-server \
  --model /home/safiyu/models/granite-4.2-8b-Q4_K_M.gguf \
  --port 9001 \
  --host 127.0.0.1 \
  --ctx-size 16384 \
  -ngl 99 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --reasoning off \
  --no-warmup \
  --threads 8 \
  --parallel 1 "$@"
