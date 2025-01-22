import json
import os
import math
import faulthandler
import argparse
import asyncio
import openai
from tqdm import tqdm

from openai import AsyncAzureOpenAI, OpenAIError, AsyncOpenAI
from utils.eval_utils import get_acc_async
from dotenv import load_dotenv
load_dotenv()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="outputs/validation_cot")
    parser.add_argument('--eval_dir', type=str, default="processed_outputs")
    args = parser.parse_args()

    subdir = os.path.basename(args.output_dir)
    os.makedirs(os.path.join(args.eval_dir, subdir), exist_ok=True)

    for output_file in os.listdir(args.output_dir):
        if not output_file.endswith(".json"):
            continue

        examples_path = os.path.join(args.output_dir, output_file)
        examples = json.load(open(examples_path))
        
        eval_file = os.path.join(args.eval_dir, subdir, output_file)
        
        # Skip if it's already evaluated
        if os.path.exists(eval_file):
            eval_results = json.load(open(eval_file))
            if (len(eval_results) == len(examples)
                and eval_results and eval_results[0]["response"] == examples[0]["response"]):
                print(f"Skipping {output_file}")
                continue
                
        # client = AsyncAzureOpenAI(
        #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        #     api_version="2024-12-01-preview",
        #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        # )
        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        accuracy, outputs = await get_acc_async(examples, client)
        json.dump(outputs, open(eval_file, "w"), indent=4, ensure_ascii=False)
        
        print(f"Accuracy of {output_file}: {accuracy}")

if __name__ == "__main__":
    asyncio.run(main())
