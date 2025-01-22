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
  "microsoft/Phi-3.5-vision-instruct:16"
  "llava-hf/LLaVA-NeXT-Video-7B-hf:16"
  "llava-hf/LLaVA-NeXT-Video-34B-hf:8"
  "unsloth/Llama-3.2-11B-Vision-Instruct:8"
  "mistral-community/pixtral-12b:8"
  "deepseek-ai/deepseek-vl2-tiny:2"
  "deepseek-ai/deepseek-vl2-small:2"
  "deepseek-ai/deepseek-vl2:2"
  "rhymes-ai/Aria-Chat:8"
  "OpenGVLab/InternVL2-8B:4"
  "OpenGVLab/InternVL2_5-8B:4"
  "OpenGVLab/InternVL2_5-38B:4"
  "HuggingFaceM4/Idefics3-8B-Llama3:4"
  "h2oai/h2ovl-mississippi-2b:4"
  "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit:8"
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