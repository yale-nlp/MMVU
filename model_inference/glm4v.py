from utils.api_utils import generate_from_openai_chat_completion_single
from utils.video_process import read_video
from utils.prepare_input import prepare_qa_inputs
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
from zhipuai import ZhipuAI
import os
import json
from tqdm import tqdm
import time
from dotenv import load_dotenv
load_dotenv()
# import logging 
from tqdm import tqdm
import asyncio

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    
    assert "glm-4v" in model_name, "Invalid model name"

    client = ZhipuAI(api_key=os.environ.get("ZHIPU_API_KEY"))
    messages = prepare_qa_inputs(model_name, queries, total_frames, prompt=prompt)

    responses = []
    for message in tqdm(messages):
        for _ in range(3):
            try:
                response = client.chat.completions.create(
                    model = model_name,
                    messages=message, 
                    max_tokens=MAX_TOKENS,
                    temperature=GENERATION_TEMPERATURE,
                )
                responses.append(response.choices[0].message.content)
                break
            except Exception as e:
                print(e)
                time.sleep(1)    

    assert len(queries) == len(responses)
    for query, response in zip(queries, responses):
        query["response"] = response
        print(response)
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)