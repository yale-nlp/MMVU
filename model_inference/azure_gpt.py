from utils.api_utils import generate_from_openai_chat_completion
from utils.video_process import read_video
from utils.prepare_input import prepare_qa_inputs
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
from openai import AsyncAzureOpenAI
import os
import json
from tqdm import tqdm
import time
from dotenv import load_dotenv
load_dotenv()
# import logging 
import asyncio

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):

    client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-07-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    if model_name == "gpt-4o-mini":
        base_rate = 200
    else:
        base_rate = 100
        
    messages = prepare_qa_inputs(model_name, queries, total_frames, prompt=prompt)

    responses = asyncio.run(
        generate_from_openai_chat_completion(
            client,
            messages=messages, 
            engine_name=model_name,
            max_tokens=MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            requests_per_minute = int(base_rate / (total_frames**0.5)),
            n = n
        )
    )

    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)