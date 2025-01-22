from utils.video_process import read_video, download_video, prepare_base64frames, prepare_gemini_video_input
from utils.constant import COT_PROMPT
import google.generativeai as genai
import requests
import os
import time
from tqdm import tqdm
import hashlib
import base64

def prepare_qa_text_input(model_name, query, prompt):
    question_type = query["question_type"]
    os.makedirs("cache", exist_ok=True)
    if question_type == "multiple-choice":
        optionized_list = [f"{key}: {value}" for i, (key, value) in enumerate(query['choices'].items())]
        optionized_str = "\n".join(optionized_list)

        user_prompt = prompt[question_type]
        qa_text_prompt = user_prompt.substitute(question=query['question'], optionized_str=optionized_str)
    elif question_type == "open-ended":
        user_prompt = prompt[question_type]
        qa_text_prompt = user_prompt.substitute(question=query['question'])
    else:
        raise ValueError(f"Invalid question type: {question_type}")
    
    qa_text_message = {
        "type": "text",
        "text": qa_text_prompt
    }
    return qa_text_message, qa_text_prompt

def prepare_multi_image_input(model_name, video_path, total_frames, video_tmp_dir = "video_cache"):
    base64frames = prepare_base64frames(model_name, video_path, total_frames, video_tmp_dir = video_tmp_dir)

    if "claude" in model_name:
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame,
                },
            } for frame in base64frames
        ]
    # for vllm models
    elif "/" in model_name: 
        return base64frames
    else:
        return [
            {
                "type": "image_url",
                'image_url': {
                    "url": f"data:image/jpeg;base64,{frame}",
                },
            } for frame in base64frames
        ]


def prepare_qa_inputs(model_name, queries, total_frames, prompt=COT_PROMPT):
    messages = []
    for query in tqdm(queries):
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        if total_frames >= 1:
            vision_input = prepare_multi_image_input(model_name, query['video'], total_frames)

            prompt_message = [
                {
                    "role": "user",
                    "content": vision_input + [qa_text_message],
                },
            ]
        elif total_frames == 0:
            prompt_message = [
                {
                    "role": "user",
                    "content": [qa_text_message],
                },
            ]
        elif total_frames == -1:
            if "gemini" in model_name:
                video_file = prepare_gemini_video_input(query['video'])
                prompt_message = [video_file, qa_text_prompt]
            elif model_name in ["GLM-4V-Plus-0111","glm-4v-plus", "glm-4v"]:
                video_url = query['video']
                prompt_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": qa_text_prompt},
                            {"type": "video_url", "video_url": {"url": video_url}}
                        ] 
                    }
                ]

        messages.append(prompt_message)
    return messages
