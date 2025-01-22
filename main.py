from tqdm import tqdm
import json
import argparse
import os
import sys
from utils.constant import COT_PROMPT, DO_PROMPT
from utils.video_process import download_video
from transformers.utils import logging
logging.set_verbosity_error() 

def main(
    model_name: str, 
    prompt: str, 
    queries: list, 
    total_frames: int, 
    output_path: str, 
    n: int=1)-> None:
    if "gpt-4o" in model_name:
        from model_inference.azure_gpt import generate_response
    elif "gemini" in model_name or "grok" in model_name or "glm" in model_name:
        from model_inference.openai_compatible import generate_response
    elif "claude" in model_name:
        from model_inference.claude import generate_response
    elif model_name in json.load(open("model_inference/vllm_model_list.json")):
        from model_inference.vllm_video_inference import generate_response
    elif "InternVideo2" in model_name:
        from model_inference.internvideo import generate_response
    elif "VideoLLaMA2" in model_name:
        from model_inference.damollama_video import generate_response
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    generate_response(model_name=model_name,
                    prompt=prompt,
                    queries=queries, 
                    total_frames=total_frames, 
                    output_path=output_path,
                    n = n)
        
prompt_dict = {
    "cot": COT_PROMPT,
    "direct-output": DO_PROMPT,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--prompt', type=str, default="cot")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--total_frames', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument("--api_base",type=str,default="")
    parser.add_argument("--max_num",type=int,default=5)
    parser.add_argument("--n",type=int,default=1)
    parser.add_argument("--overwrite",action="store_true", default=False)

    args = parser.parse_args()
    model_name = args.model
    total_frames = args.total_frames # total_frames

    try:
        prompt = prompt_dict[args.prompt]
    except KeyError:
        print("Invalid prompt")
        sys.exit(1)


    os.makedirs(args.output_dir, exist_ok=True)
        
    data_path_suffix = args.data_path.split("/")[-1].split(".")[0]
    output_dir = os.path.join(args.output_dir, f"{data_path_suffix}_{args.prompt}")
    os.makedirs(output_dir, exist_ok=True)
    
    output_name = model_name.split("/")[-1]
    if total_frames == -1:
        output_path = os.path.join(output_dir, f"{output_name}_1fps.json")
    else:
        output_path = os.path.join(output_dir, f"{output_name}_{total_frames}frame.json")


    queries = json.load(open(args.data_path, "r"))
    if args.max_num > 0:
        queries = queries[:args.max_num]
    
    # download videos
    for query in queries:
        download_video(query['video'])

    if os.path.exists(output_path) and args.overwrite:
        orig_output = json.load(open(output_path, "r"))
        if len(orig_output) >= len(queries):
            print(f"Output file {output_path} already exists. Skipping.")
            sys.exit(0)
        else:
            print(f"Overwrite {output_path}")

    print(f"=========Running {args.model}=========\n")
    main(
        model_name = args.model, 
        prompt = prompt, 
        queries = queries, 
        total_frames = total_frames, 
        output_path = output_path, 
        n = args.n)