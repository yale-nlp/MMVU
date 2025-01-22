import base64
import numpy as np
from typing import Dict
import random

import asyncio
import logging
import os, json
from typing import Any
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
import random
from time import sleep

import aiolimiter

import openai
from openai import AsyncOpenAI, OpenAIError, AzureOpenAI, AsyncAzureOpenAI
from anthropic import AsyncAnthropic

async def _throttled_openai_chat_completion_acreate_single(
    client: AsyncOpenAI,
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    json_format: bool = False,
    n: int = 1,
):
    async with limiter:
        for _ in range(10):
            try:
                if json_format:
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        response_format={"type": "json_object"},
                    )
                else:
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
            except openai.RateLimitError as e:
                print("Rate limit exceeded, retrying...")
                await asyncio.sleep(random.randint(10, 20)) 
            except openai.BadRequestError as e:
                print(e)
                return None
            except OpenAIError as e:
                print(e)
                await asyncio.sleep(random.randint(5, 10))
        return None

async def generate_from_openai_chat_completion_single(
    client,
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 100,
    json_format: bool = False,
    n: int = 1,
):
    # https://chat.openai.com/share/09154613-5f66-4c74-828b-7bd9384c2168
    delay = 60.0 / requests_per_minute
    limiter = aiolimiter.AsyncLimiter(1, delay)
    async_responses = [
        _throttled_openai_chat_completion_acreate_single(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            json_format=json_format,
            n=n,
        )
        for message in messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    for response in responses:
        if n == 1:
            if json_format:
                outputs.append(json.loads(response.choices[0].message.content))
            else:
                outputs.append(response.choices[0].message.content)
        else:
            if json_format:
                outputs.append([json.loads(response.choices[i].message.content) for i in range(n)])
            else:
                outputs.append([response.choices[i].message.content for i in range(n)])
    return outputs

async def _throttled_openai_chat_completion_acreate(
    client: AsyncOpenAI,
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    json_format: bool = False,
    n: int = 1,
):
    async with limiter:
        for _ in range(10):
            try:
                if json_format:
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        n=n,
                        response_format={"type": "json_object"},
                    )
                else:
                    return await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        n=n,
                    )
            except openai.RateLimitError as e:
                print("Rate limit exceeded, retrying...")
                await asyncio.sleep(random.randint(10, 20))  # 增加重试等待时间
            except openai.BadRequestError as e:
                print(e)
                return None
            except OpenAIError as e:
                print(e)
                await asyncio.sleep(random.randint(5, 10))
        return None

async def generate_from_openai_chat_completion(
    client,
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 150,
    json_format: bool = False,
    n: int = 1,
):
    delay = 60.0 / requests_per_minute
    limiter = aiolimiter.AsyncLimiter(1, delay)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            json_format=json_format,
            n=n,
        )
        for message in messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    for response in responses:
        if n == 1:
            try:
                if json_format:
                    outputs.append(json.loads(response.choices[0].message.content))
                else:
                    outputs.append(response.choices[0].message.content)
            except Exception as e:
                print(e)
                outputs.append("")
        else:
            if json_format:
                outputs.append([json.loads(response.choices[i].message.content) for i in range(n)])
            else:
                outputs.append([response.choices[i].message.content for i in range(n)])
    return outputs

async def _throttled_claude_chat_completion_acreate(
    client: AsyncAnthropic,
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
):
    async with limiter:
        try:
            return await client.messages.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        except:
            return None

async def generate_from_claude_chat_completion(
    client,
    messages,
    engine_name: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 150,
    n: int = 1,
):
    # https://chat.openai.com/share/09154613-5f66-4c74-828b-7bd9384c2168
    delay = 60.0 / requests_per_minute
    limiter = aiolimiter.AsyncLimiter(1, delay)
    
    n_messages = []
    for message in messages:
        for _ in range(n):
            n_messages.append(message)

    async_responses = [
        _throttled_claude_chat_completion_acreate(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in n_messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    if n == 1:
        try:
            for response in responses:
                outputs.append(response.content[0].text)
        except:
            outputs.append("")
    else:
        idx = 0
        for response in responses:
            if idx % n == 0:
                outputs.append([])
            idx += 1
            outputs[-1].append(response.content[0].text)

    return outputs


async def _throttled_gemini_chat_completion_acreate(
    model,
    messages,
    generation_config,
    safety_settings,
    limiter: aiolimiter.AsyncLimiter,
):
    async with limiter:
        for _ in range(10):
            try:
                return await model.generate_content_async(
                    messages,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
            except Exception as e:
                print(e)
                await asyncio.sleep(random.randint(5, 10))

        return None

async def generate_from_gemini_chat_completion(
    model,
    messages,
    generation_config,
    safety_settings,
    requests_per_minute: int = 500,
    n: int = 1,
):
    # https://chat.openai.com/share/09154613-5f66-4c74-828b-7bd9384c2168
    if "flash" in model.model_name:
        requests_per_minute = 400
    elif "pro" in model.model_name:
        requests_per_minute = 100

    delay = 60.0 / requests_per_minute
    limiter = aiolimiter.AsyncLimiter(1, delay)
    
    n_messages = []
    for message in messages:
        for _ in range(n):
            n_messages.append(message)

    async_responses = [
        _throttled_gemini_chat_completion_acreate(
            model=model,
            messages=message,
            generation_config=generation_config,
            safety_settings=safety_settings,
            limiter=limiter,
        )
        for message in n_messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    errors = 0
    if n == 1:
        for response in responses:
            try:
                response.resolve()
                outputs.append(response.text)
            except:
                errors += 1
                outputs.append("Error")
    else:
        idx = 0
        for response in responses:
            if idx % n == 0:
                outputs.append([])
            idx += 1
            response.resolve()
            outputs[-1].append(response.text)

    print(f"Error number: {errors}")
    return outputs

async def _throttled_openai_chat_completion_acreate_structure(
    client,
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    structured_format
):
    async with limiter:
        for _ in range(10):
            try:
                return await client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    response_format=structured_format,
                )
                
            except openai.RateLimitError as e:
                print("Rate limit exceeded, retrying...")
                await asyncio.sleep(random.randint(10, 20)) 
            except openai.BadRequestError as e:
                print(e)
                return None
            except OpenAIError as e:
                print(e)
                await asyncio.sleep(random.randint(5, 10))
        return None

async def generate_from_openai_chat_completion_structured_format(
    client,
    messages,
    engine_name: str,
    structured_format,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    requests_per_minute: int = 100,
):
    # https://chat.openai.com/share/09154613-5f66-4c74-828b-7bd9384c2168
    delay = 60.0 / requests_per_minute
    limiter = aiolimiter.AsyncLimiter(1, delay)
    async_responses = [
        _throttled_openai_chat_completion_acreate_structure(
            client,
            model=engine_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            structured_format=structured_format
        )
        for message in messages
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses)
    
    outputs = []
    for response in responses:
        try:
            response = response.choices[0].message.parsed
        except:
            response = None
        outputs.append(response)
    return outputs