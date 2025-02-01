from utils.vlm_prepare_input import *
import json
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT

model_example_map = {
    "llava-hf/LLaVA-NeXT-Video-7B-hf": prepare_llava_next_video_inputs,
    "llava-hf/LLaVA-NeXT-Video-34B-hf": prepare_llava_next_video_inputs,
    "Qwen/Qwen2-VL-7B-Instruct": prepare_qwen2_inputs, 
    "Qwen/Qwen2-VL-2B-Instruct": prepare_qwen2_inputs,
    "Qwen/Qwen2-VL-72B-Instruct-AWQ": prepare_qwen2_inputs, 
    "microsoft/Phi-3.5-vision-instruct": prepare_phi3v_inputs, 
    "OpenGVLab/InternVL2-8B": prepare_general_vlm_inputs,
    "OpenGVLab/InternVL2_5-78B-AWQ": prepare_general_vlm_inputs,
    "OpenGVLab/InternVL2_5-8B": prepare_general_vlm_inputs,
    "OpenGVLab/InternVL2_5-38B":
    prepare_general_vlm_inputs,
    "mistral-community/pixtral-12b": prepare_pixtral_inputs,
    "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf": prepare_llava_onevision_inputs,
    "unsloth/Llama-3.2-11B-Vision-Instruct": prepare_mllama_inputs,
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit": prepare_mllama_inputs,
    "h2oai/h2ovl-mississippi-2b": prepare_general_vlm_inputs,
    "nvidia/NVLM-D-72B": prepare_general_vlm_inputs,
    "HuggingFaceM4/Idefics3-8B-Llama3": prepare_general_vlm_inputs,
    "deepseek-ai/deepseek-vl2": prepare_deepseek_vl2_inputs,
    "deepseek-ai/deepseek-vl2-tiny": prepare_deepseek_vl2_inputs,
    "deepseek-ai/deepseek-vl2-small": prepare_deepseek_vl2_inputs,
    "rhymes-ai/Aria-Chat": prepare_aria_inputs,
    "Qwen/Qwen2.5-VL-3B-Instruct": prepare_qwen2_inputs,
    "Qwen/Qwen2.5-VL-7B-Instruct": prepare_qwen2_inputs,
}

vllm_model_list = list(model_example_map.keys())
json.dump(vllm_model_list, open("model_inference/vllm_model_list.json", "w"), indent=4, ensure_ascii=False)

def generate_vlm_response(model_name, 
                       queries, 
                       total_frames, 
                       prompt=COT_PROMPT, 
                       temperature: float=1, 
                       max_tokens: int=1024):
    inputs, llm, sampling_params = model_example_map[model_name](model_name, queries, prompt, total_frames, temperature, max_tokens)
    
    responses = llm.generate(inputs, sampling_params=sampling_params)
    responses = [response.outputs[0].text for response in responses]
    return responses

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1,
                    temperature: float=GENERATION_TEMPERATURE,
                    max_tokens: int=MAX_TOKENS):
    if model_name not in model_example_map:
        raise ValueError(f"Model type {model_name} is not supported.")
    responses = generate_vlm_response(model_name, 
                                      queries, 
                                      prompt=prompt, 
                                      total_frames=total_frames, 
                                      temperature=temperature, 
                                      max_tokens=max_tokens)
    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
