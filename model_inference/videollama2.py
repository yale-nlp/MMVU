import os, sys
sys.path.append('VideoLLaMA2')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
from utils.video_process import download_video
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
import json
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input
import hashlib
import requests
from tqdm import tqdm

def generate_by_videollama2(model_name, 
                            queries, 
                            prompt, 
                            total_frames, 
                            temperature, 
                            max_tokens):
    inputs = []
    responses = []
    model, processor, tokenizer = model_init(model_name)
    modal = 'video'
    for query in tqdm(queries):
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        text_input = f"{qa_text_prompt}"
        video_path, _ = download_video(query['video'])

        input = {
                "prompt": text_input,
                "multi_modal_data": {
                    "video": video_path
                },
            }
        response = mm_infer(processor[modal](input['multi_modal_data']['video']),  input['prompt'], model=model, tokenizer=tokenizer, do_sample=False, modal=modal)
        responses.append(response)
    return responses

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    responses = generate_by_videollama2(model_name, 
                                      queries, 
                                      prompt=prompt, 
                                      total_frames=total_frames, 
                                      temperature=GENERATION_TEMPERATURE, 
                                      max_tokens=MAX_TOKENS)
    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)
