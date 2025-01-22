from transformers import AutoTokenizer
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser

from argparse import Namespace
from typing import List
import torch
from transformers import AutoProcessor, AutoTokenizer

from vllm.assets.image import ImageAsset
from utils.video_process import video_to_ndarrays_fps, video_to_ndarrays, download_video

from vllm.multimodal.utils import fetch_image
import os
import hashlib
import base64
import requests
from tqdm import tqdm
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input

def prepare_llava_next_video_inputs(model_name, 
                        queries, 
                        prompt,
                        total_frames: int=-1, 
                        temperature: float=1,
                        max_tokens: int=1024):
    responses = []
    stop_token_ids = None
    sampling_params = SamplingParams(temperature=temperature,
                                    max_tokens=max_tokens,
                                    stop_token_ids=stop_token_ids)
    if model_name == "llava-hf/LLaVA-NeXT-Video-7B-hf":
        max_model_len = 8192
    elif model_name == "llava-hf/LLaVA-NeXT-Video-34B-hf":
        max_model_len = 4096
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    llm = LLM(model=model_name,
             max_model_len=max_model_len,
             limit_mm_per_prompt={"video": 1},
             tensor_parallel_size=min(torch.cuda.device_count(),4),
             )
    inputs = []
    for query in tqdm(queries, desc="Prepare model inputs"):
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        text_input = f"USER: <video>\n{qa_text_prompt} ASSISTANT:"
        video_path, _ = download_video(query['video'])
        
        # does not support dynamic frame number in batch inference
        video_data = video_to_ndarrays(path=video_path, num_frames=total_frames)

        input = {
                "prompt": text_input,
                "multi_modal_data": {
                    "video": video_data
                },
            }
        
        inputs.append(input)
    
    return inputs, llm, sampling_params

def prepare_qwen2_inputs(model_name, 
                queries,
                prompt,
                total_frames: int=-1, 
                temperature: float=1,
                max_tokens: int=1024):
    inputs = []
    llm = LLM(model=model_name,
              tensor_parallel_size=min(torch.cuda.device_count(),4),
              limit_mm_per_prompt={"video": 1})
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=None)
    
    for query in tqdm(queries, desc="Prepare model inputs"):
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        video_path, _ = download_video(query['video'])
        if total_frames == -1:
            video_data = video_to_ndarrays_fps(path=video_path, fps=1, max_frames=64)
        else:
            video_data = video_to_ndarrays(path=video_path, num_frames=total_frames)
        
        processor = AutoProcessor.from_pretrained(model_name, min_pixels = 256*28*28, max_pixels = 1280*28*28)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {"type": "text", "text": qa_text_prompt},
                ],
            }
        ]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input = {
                "prompt": text_input,
                "multi_modal_data": {
                    "video": video_data
                },
            }
        inputs.append(input)
    
    return inputs, llm, sampling_params

def prepare_phi3v_inputs(model_name, 
                queries,
                prompt,
                total_frames, 
                temperature: float=1,
                max_tokens: int=1024):
    inputs = []
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": total_frames},
        tensor_parallel_size=min(torch.cuda.device_count(),4),
    )
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=None)
    
    for query in tqdm(queries, desc="Prepare model inputs"):
        vision_input_base64 = prepare_multi_image_input(model_name, query['video'], total_frames)
        vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        placeholders = "\n".join(f"<|image_{i}|>"
                                for i, _ in enumerate(vision_input, start=1))
        text_input = f"<|user|>\n{placeholders}\n{qa_text_prompt}<|end|>\n<|assistant|>\n"

        
        input = {
                "prompt": text_input,
                "multi_modal_data": {
                    "image": vision_input
                },
            }
        inputs.append(input)

    return inputs, llm, sampling_params

def prepare_deepseek_vl2_inputs(model_name, 
                queries,
                prompt,
                total_frames, 
                temperature: float=1,
                max_tokens: int=1024):
    inputs = []
    llm = LLM(model=model_name,
        max_model_len=4096,
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={"image": total_frames},
        tensor_parallel_size=min(torch.cuda.device_count(),4),
        )

    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=None)
    
    for query in tqdm(queries, desc="Prepare model inputs"):
        vision_input_base64 = prepare_multi_image_input(model_name, query['video'], total_frames)
        vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
        
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        placeholder = "".join(f"image_{i}:<image>\n"
                          for i, _ in enumerate(vision_input, start=1))
        text_input = f"<|User|>: {placeholder}{qa_text_prompt}\n\n<|Assistant|>:"
        input = {
                "prompt": text_input,
                "multi_modal_data": {
                    "image": vision_input
                },
            }
        inputs.append(input)

    return inputs, llm, sampling_params

def prepare_aria_inputs(model_name, 
                queries,
                prompt,
                total_frames, 
                temperature: float=1,
                max_tokens: int=1024):
    inputs = []
    llm = LLM(model=model_name,
            tokenizer_mode="slow",
            trust_remote_code=True,
            limit_mm_per_prompt={"image": total_frames},
            tensor_parallel_size=min(torch.cuda.device_count(),4),
            )
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    stop_token_ids = [93532, 93653, 944, 93421, 1019, 93653, 93519,17]
    
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)
    
    for query in tqdm(queries, desc="Prepare model inputs"):
        vision_input_base64 = prepare_multi_image_input(model_name, query['video'], total_frames)
        vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
        
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        placeholder = "<fim_prefix><|img|><fim_suffix>\n" * len(vision_input)
        text_input = f"<|im_start|>user\n{placeholder}{qa_text_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        input = {
                "prompt": text_input,
                "multi_modal_data": {
                    "image": vision_input
                },
            }
        inputs.append(input)

    return inputs, llm, sampling_params

def prepare_general_vlm_inputs(model_name, 
                queries,
                prompt,
                total_frames, 
                temperature: float=1,
                max_tokens: int=1024):
    inputs = []

    if "h2oai" in model_name:
        max_model_len=8192
    else:
        max_model_len=16384
    llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": total_frames},
            tensor_parallel_size=min(torch.cuda.device_count(),4),
        )

    for query in tqdm(queries, desc="Prepare model inputs"):
        vision_input_base64 = prepare_multi_image_input(model_name, query['video'], total_frames)
        vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
        
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        placeholders = "\n".join(f"Image-{i}: <image>\n"
                                for i, _ in enumerate(vision_input, start=1))
        messages = [{'role': 'user', 'content': f"{placeholders}\n{qa_text_prompt}"}]
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                trust_remote_code=True)
        text_input = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
        
        input = {
                "prompt": text_input,
                "multi_modal_data": {
                    "image": vision_input
                },
            }
        inputs.append(input)

    if "h2oai" in model_name:
        stop_token_ids = [tokenizer.eos_token_id]
    else:
        stop_token_ids = None
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)

    return inputs, llm, sampling_params

def prepare_pixtral_inputs(model_name, 
                queries,
                prompt,
                total_frames, 
                temperature: float=1,
                max_tokens: int=1024):
    inputs = []
    stop_token_ids = None
    llm = LLM(model=model_name, 
            max_model_len=8192,
            max_num_seqs=2,
            limit_mm_per_prompt={"image": total_frames}, 
            tensor_parallel_size=min(torch.cuda.device_count(),4))
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)     
    inputs = []
    for query in tqdm(queries, desc="Prepare model inputs"):
        vision_inputs = prepare_multi_image_input(model_name, query['video'], total_frames)
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        placeholders = "[IMG]" * len(vision_inputs)
        cur_input = f"<s>[INST]{qa_text_prompt}\n{placeholders}[/INST]"
        inputs.append(cur_input)
        
    return inputs, llm, sampling_params

def prepare_mllama_inputs(model_name, 
                queries,
                prompt,
                total_frames, 
                temperature: float=1,
                max_tokens: int=1024):
    inputs = []
    if "11B" in model_name:
        llm = LLM(model=model_name,
            limit_mm_per_prompt={"image":total_frames},
            max_model_len=8192,
            max_num_seqs=2,
            enforce_eager=True,
            trust_remote_code=True,
            tensor_parallel_size=min(torch.cuda.device_count(),4),
        )
    else:
        llm = LLM(model=model_name,
            limit_mm_per_prompt={"image":total_frames},
            max_model_len=8192,
            max_num_seqs=2,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            enforce_eager=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.95
        )
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=[128001,128008,128009])

    for query in tqdm(queries, desc="Prepare model inputs"):
        vision_input_base64 = prepare_multi_image_input(model_name, query['video'], total_frames)
        vision_inputs = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        placeholders = "<|image|>" * len(vision_inputs)
        messages = [{'role': 'user', 'content': f"{placeholders}\n{qa_text_prompt}"}]
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
        input_prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

        input = {
                "prompt": input_prompt,
                "multi_modal_data": {
                    "image": vision_inputs
                },
            }
        inputs.append(input)

    return inputs, llm, sampling_params

def prepare_llava_onevision_inputs(model_name, 
                queries,
                prompt,
                total_frames, 
                temperature: float=1,
                max_tokens: int=1024):
    inputs = []
    for query in tqdm(queries, desc="Prepare model inputs"):
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        input_prompt = f"<|im_start|>user <video>\n{qa_text_prompt}<|im_end|> \
        <|im_start|>assistant\n"
        
        video_path, _ = download_video(query['video'])
        if total_frames == -1:
            video_data = video_to_ndarrays_fps(path=video_path, fps=1, max_frames=64)
        else:
            video_data = video_to_ndarrays(path=video_path, num_frames=total_frames)

        input = {
                "prompt": input_prompt,
                "multi_modal_data": {
                    "video": video_data
                },
            }
        inputs.append(input)
    
    llm = LLM(model=model_name,
                max_model_len=32768,
                limit_mm_per_prompt={"video": 1},
                tensor_parallel_size=min(torch.cuda.device_count(),4))  
    stop_token_ids = None  
    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)    

    return inputs, llm, sampling_params
