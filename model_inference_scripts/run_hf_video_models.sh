#!/bin/bash
export VLLM_LOGGING_LEVEL=ERROR

# Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Common parameters
MAX_NUM=-1
TOTAL_FRAMES=-1
DATA_PATHS=(
  # "data/test.json"
  "data/validation.json"
)

OPTIONS="--overwrite"

# Model configurations (conda environment and model name)
# videollama2 requirements: 
# internvideo requirements:
MODELS=(
  "internvideo:OpenGVLab/InternVideo2-Chat-8B"
  "videollama2:DAMO-NLP-SG/VideoLLaMA2-7B-16F"
  "videollama2:DAMO-NLP-SG/VideoLLaMA2.1-7B-16F"
  "videollama2:DAMO-NLP-SG/VideoLLaMA3-7B"
  "videollama2:DAMO-NLP-SG/VideoLLaMA3-2B"
)

PROMPTS=(
    "cot"
    "direct-output"
)

for DATA_PATH in "${DATA_PATHS[@]}"; do
    for PROMPT in "${PROMPTS[@]}"; do
        for ENTRY in "${MODELS[@]}"; do
            IFS=":" read -r ENVIRONMENT MODEL <<< "$ENTRY"
            conda activate "$ENVIRONMENT"
            python main.py --model "$MODEL" \
                        --prompt "$PROMPT" \
                        --max_num "$MAX_NUM" \
                        --total_frames "$TOTAL_FRAMES" \
                        --data_path "$DATA_PATH" \
                        $OPTIONS
        done
    done
done