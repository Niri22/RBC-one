# llm_judge_helpers.py
import re
import json
import torch
import asyncio
from asyncio import Semaphore
from transformers import AutoTokenizer
from openai import AsyncOpenAI


def extract_qa(conv_text: str):
    q  = re.search(r"Question:(.*?)(?:Answer 1:|$)", conv_text, re.DOTALL)
    a1 = re.search(r"Answer 1:(.*?)(?:Answer 2:|$)", conv_text, re.DOTALL)
    a2 = re.search(r"Answer 2:(.*)", conv_text, re.DOTALL)
    return (
        q.group(1).strip() if q else "",
        a1.group(1).strip() if a1 else "",
        a2.group(1).strip() if a2 else "",
    )


def safe_json_loads(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def run_local_inference(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()


async def judge_with_openai(
    client: AsyncOpenAI,
    semaphore: Semaphore,
    judge_model: str,
    q: str,
    a1: str,
    a2: str,
    base_out: str,
    dpo_out: str,
):
    query = f"""
You are an impartial expert evaluator.

Evaluate the following two model judgments independently on a scale from 0 to 10
based on:
- coherence
- accuracy
- coverage
- overall quality

Question:
{q}

Answer 1:
{a1}

Answer 2:
{a2}

Model A (Base model) judgment:
{base_out}

Model B (DPO-trained model) judgment:
{dpo_out}

Respond ONLY in valid JSON:

{{
  "base_score": number,
  "dpo_score": number,
  "reason": "short explanation"
}}
"""

    async with semaphore:
        r = await client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": query}],
            temperature=0,
        )

    raw = r.choices[0].message.content.strip()
    return safe_json_loads(raw)
