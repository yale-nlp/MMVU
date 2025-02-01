#!/bin/bash

# Environment variable setup
export VLLM_CONFIGURE_LOGGING=0
export VLLM_LOGGING_LEVEL=ERROR
export PYTHONWARNINGS="ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false

# Common parameters
MAX_NUM=-1
DATA_PATHS=(
  "data/validation.json"
  # "data/test.json"
)
OPTIONS="--overwrite"

# Model configurations (model name and corresponding total_frames)
MODELS=(
  "gemini-1.5-flash:32"
  "gemini-1.5-pro:32"
  "gemini-2.0-flash-exp:32"
  "gemini-2.0-flash-thinking-exp-1219:32"
  "gpt-4o:32"
  "gpt-4o-mini:32"
  "glm-4v-plus-0111:-1" # use video input
  "claude-3-5-sonnet-20241022:32"
  "grok-2-vision-latest:16"
)

PROMPTS=(
    "cot"
    "direct-output"
)

# Execute the script for each model
for DATA_PATH in "${DATA_PATHS[@]}"; do
  for PROMPT in "${PROMPTS[@]}"; do
    for ENTRY in "${MODELS[@]}"; do
      IFS=":" read -r MODEL TOTAL_FRAMES <<< "$ENTRY"
      python main.py --model "$MODEL" \
                     --prompt "$PROMPT" \
                     --total_frames "$TOTAL_FRAMES" \
                     --max_num "$MAX_NUM" \
                     --data_path "$DATA_PATH" \
                     $OPTIONS
    done
  done
done