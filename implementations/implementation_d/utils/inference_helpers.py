# inference_helpers.py

import os
import re
import json
import glob
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import jsonlines
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def clean_json_output(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _suffix_num(path):
    try:
        return int(Path(path).stem.split("_")[-1])
    except Exception:
        return -1


def save_checkpoint(
    scenes: List[dict],
    task_name: str,
    checkpoint_dir: str,
    step: int,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_{task_name}_{step}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2, ensure_ascii=False)
    print(f"[ckpt] saved → {path}")


def load_checkpoint(task_name: str, checkpoint_dir: str):
    pattern = os.path.join(checkpoint_dir, f"ckpt_{task_name}_*.json")
    files = glob.glob(pattern)
    if not files:
        return [], -1

    latest = max(files, key=_suffix_num)
    with open(latest, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    last_idx = max((s.get("prompt_idx", -1) for s in scenes), default=-1)
    print(f"[ckpt] loaded ← {latest} (last_idx={last_idx})")
    return scenes, last_idx


def apply_chat_template(prompt: str, tokenizer: AutoTokenizer) -> str:
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
    except Exception:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )


def prepare_record(rec: dict, tokenizer: AutoTokenizer):
    meta = {k: rec[k] for k in rec if k != "prompt"}
    prompt = apply_chat_template(rec["prompt"], tokenizer)
    return meta, prompt


def load_disk_records(path: str, limit: int = 200):
    ds = load_from_disk(path)["train"]
    return [ds[i] for i in range(min(limit, len(ds)))]


def load_arrow_records(path: str, limit: int = 200):
    ds = load_dataset("arrow", data_files=path, split="train")
    ds = ds.select(range(min(limit, len(ds))))
    return [dict(r) for r in ds]


QA_PATTERN = re.compile(
    r"Question:\s*(.*?)\s*Answer 1:\s*(.*?)\s*Answer 2:\s*(.*)",
    re.DOTALL,
)


def build_prompt_records(
    dataset: List[dict],
    templates: Dict[str, list],
    template_key: str,
    reverse: bool = False,
):
    tpl = templates[template_key]
    records = []

    for i, it in enumerate(dataset):
        it = dict(it)
        chosen = it.get("chosen_id", it.get("chosen"))

        if not all(k in it for k in ("q", "r1", "r2")):
            m = QA_PATTERN.search(it["prompt"])
            it["q"], it["r1"], it["r2"] = m.group(1), m.group(2), m.group(3)

        prompt = tpl[0] + it["q"] + tpl[1] + it["r1"] + tpl[2] + it["r2"]
        hint = chosen if not reverse else (3 - chosen)
        prompt += tpl[3] + str(hint) + tpl[4]

        records.append({
            "prompt_idx": i,
            "prompt": prompt,
            "chosen": chosen,
            "meta": it,
        })

    return records


def run_best_of_n(
    records,
    model,
    tokenizer,
    output_path,
    checkpoint_dir,
    task_name,
    n=8,
    checkpoint_every=5,
    max_new_tokens=512,
    prompt_max_len=6400,
):
    scenes, last_idx = load_checkpoint(task_name, checkpoint_dir)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = jsonlines.open(output_path, "w")

    for i, rec in enumerate(tqdm(records, desc="Best-of-N")):
        if i <= last_idx:
            continue

        meta, prompt = prepare_record(rec, tokenizer)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=prompt_max_len,
        ).to(model.device)

        expanded = {k: v.repeat(n, 1) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **expanded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.2,
                top_p=0.95,
            )

        in_len = inputs["attention_mask"].sum(dim=1)[0].item()
        gens = [
            clean_json_output(
                tokenizer.decode(out[j, in_len:], skip_special_tokens=True)
            )
            for j in range(out.size(0))
        ]

        scenes.append({
            "prompt_idx": i,
            "prompt": prompt,
            "outputs": gens,
            "meta": meta,
        })

        if len(scenes) % checkpoint_every == 0:
            save_checkpoint(scenes, task_name, checkpoint_dir, len(scenes))

    for s in scenes:
        writer.write(s)
    writer.close()

def run_batched_inference(
    records,
    model,
    tokenizer,
    batch_size=4,
    max_new_tokens=512,
):
    """
    Deterministic batched generation.
    Returns list of {prompt_idx, prompt, output, meta}
    """
    results = []

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        prompts = [r["prompt"] for r in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=6400,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        gens = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for rec, gen in zip(batch, gens):
            results.append({
                "prompt_idx": rec["prompt_idx"],
                "prompt": rec["prompt"],
                "output": gen,
                "meta": rec["meta"],
            })

    return results

