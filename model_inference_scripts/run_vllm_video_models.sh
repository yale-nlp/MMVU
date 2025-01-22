#!/bin/bash

# Environment variable setup
export VLLM_CONFIGURE_LOGGING=0
export VLLM_LOGGING_LEVEL=ERROR
export PYTHONWARNINGS="ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false

# Common parameters
TOTAL_FRAMES=-1
MAX_NUM=-1
DATA_PATHS=(
  # "data/test.json"
  "data/validation.json"
)
OPTIONS="--overwrite"

# Models to run
MODELS=(
  "Qwen/Qwen2-VL-2B-Instruct"
  "Qwen/Qwen2-VL-7B-Instruct"
  "Qwen/Qwen2-VL-72B-Instruct-AWQ"
  "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"
)

PROMPTS=(
    "cot"
    "direct-output"
)

# Execute the script for each model
for DATA_PATH in "${DATA_PATHS[@]}"; do
  for PROMPT in "${PROMPTS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      python main.py --model "$MODEL" \
                    --prompt "$PROMPT" \
                    --total_frames "$TOTAL_FRAMES" \
                    --max_num "$MAX_NUM" \
                    --data_path "$DATA_PATH" \
                    $OPTIONS
    done
  done
done