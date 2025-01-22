from utils.api_utils import generate_from_claude_chat_completion
from utils.video_process import read_video
from utils.prepare_input import prepare_qa_inputs
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
import anthropic
import os
import time
import json 
from tqdm import tqdm
import asyncio

from dotenv import load_dotenv
load_dotenv()


def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):

    client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=10)

    messages = prepare_qa_inputs(model_name, queries, total_frames, prompt=prompt)

    responses = asyncio.run(
        generate_from_claude_chat_completion(
            client,
            messages=messages, 
            engine_name=model_name,
            max_tokens=MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            requests_per_minute = int(90 / (total_frames**0.5)) if total_frames != 0 else 200,
            n = n
        )
    )

    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)
