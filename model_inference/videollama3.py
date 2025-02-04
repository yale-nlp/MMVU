from utils.video_process import download_video
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
import json
from utils.prepare_input import prepare_qa_text_input
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def generate_by_videollama3(model_name, 
                            queries, 
                            prompt,
                            total_frames, 
                            temperature, 
                            max_tokens):
    inputs = []
    responses = []

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    for i, query in enumerate(tqdm(queries)):
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        text_input = f"{qa_text_prompt}"
        video_path, _ = download_video(query['video'])

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 180}},
                    {"type": "text", "text": text_input},
                ]
            },
        ]

        inputs = processor(
            conversation=conversation,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            max_new_tokens=2048,
        )
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        responses.append(response)

    return responses

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    responses = generate_by_videollama3(model_name, 
                                      queries, 
                                      prompt=prompt, 
                                      total_frames=total_frames, 
                                      temperature=GENERATION_TEMPERATURE, 
                                      max_tokens=MAX_TOKENS)
    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)
