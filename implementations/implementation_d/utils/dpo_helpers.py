# dpo_helpers.py
import os
import json
import random
import re
import numpy as np
from datasets import Dataset

random.seed(2021)

def safe_json_loads(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def evaluate(output_str):
    """
    Parse model outputs to extract better_answer = 1 or 2.
    Robust to malformed JSON.
    """
    output_str = output_str.strip().replace('”,', '",').replace('”', '"')

    # Case 1: fenced JSON
    if "```json" in output_str:
        match = re.search(r'```json(.*?)```', output_str, re.DOTALL)
        if match:
            data = safe_json_loads(match.group(1))
            if data:
                return data.get("better_answer") or data.get("better answer")

    # Case 2: raw JSON
    if output_str.startswith("{"):
        data = safe_json_loads(output_str)
        if data:
            return data.get("better_answer") or data.get("better answer")

    # Case 3: embedded JSON
    if "{" in output_str and "}" in output_str:
        match = re.search(r'{(.*?)}', output_str, re.DOTALL)
        if match:
            data = safe_json_loads("{" + match.group(1) + "}")
            if data:
                return data.get("better_answer") or data.get("better answer")

    # Case 4: regex fallback
    match = re.search(r'"better[_ ]answer"\s*:\s*(1|2)', output_str)
    if match:
        return int(match.group(1))

    return None

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_positive_negative(dataset):
    grouped = {}
    none_rate = []

    for item in dataset:
        prompt = item.get("prompt", "")
        meta = item.get("meta", {})
        test_id = item.get("test_id", item.get("prompt_idx"))
        correct_answer = meta.get("chosen", item.get("chosen"))
        tag = meta.get("tag", item.get("tag", "default"))

        grouped.setdefault(test_id, {"positive": [], "negative": [], "unknown": []})

        for gen_str in item.get("outputs", []):
            eval_result = evaluate(gen_str)
            sample = {
                "prompt": prompt,
                "gen": gen_str,
                "chosen": correct_answer,
                "test_id": test_id,
                "tag": tag,
            }

            if eval_result == correct_answer:
                grouped[test_id]["positive"].append(sample)
                none_rate.append(0)
            elif eval_result is None:
                grouped[test_id]["unknown"].append(sample)
                none_rate.append(1)
            else:
                grouped[test_id]["negative"].append(sample)
                none_rate.append(0)

    print(f"Unknown rate: {np.mean(none_rate) if none_rate else 0:.3f}")
    return grouped

def construct_dpo_pairs(guide_split, guide_rev_split, output_split):
    pairs = {
        "conversations": [],
        "chosen": [],
        "rejected": [],
        "pair_type": [],
        "test_id": [],
        "tag": [],
        "chosen_id": [],
    }

    stats = {
        "best_of_n": 0,
        "best_of_n_positive2unknown": 0,
        "preamble": 0,
        "preamble2unknown": 0,
        "num_lost": 0,
    }

    for test_id, data in output_split.items():
        pos, neg, unk = data["positive"], data["negative"], data["unknown"]
        random.shuffle(pos)
        random.shuffle(neg)
        random.shuffle(unk)

        # Best-of-N
        if pos and neg:
            for i in range(min(len(pos), len(neg), 3)):
                pairs["conversations"].append([{"from": "human", "value": pos[i]["prompt"]}])
                pairs["chosen"].append(pos[i]["gen"])
                pairs["rejected"].append(neg[i]["gen"])
                pairs["pair_type"].append("best_of_n")
                pairs["test_id"].append(test_id)
                pairs["tag"].append(pos[i]["tag"])
                pairs["chosen_id"].append(pos[i]["chosen"])
                stats["best_of_n"] += 1

        elif pos and unk:
            pairs["conversations"].append([{"from": "human", "value": pos[0]["prompt"]}])
            pairs["chosen"].append(pos[0]["gen"])
            pairs["rejected"].append(unk[0]["gen"])
            pairs["pair_type"].append("best_of_n_positive2unknown")
            pairs["test_id"].append(test_id)
            pairs["tag"].append(pos[0]["tag"])
            pairs["chosen_id"].append(pos[0]["chosen"])
            stats["best_of_n_positive2unknown"] += 1

        # Preamble (guide vs guide_reverse)
        if test_id in guide_split and test_id in guide_rev_split:
            g_pos = guide_split[test_id]["positive"]
            g_rev_neg = guide_rev_split[test_id]["negative"]

            if g_pos and g_rev_neg:
                pairs["conversations"].append([{"from": "human", "value": g_pos[0]["prompt"]}])
                pairs["chosen"].append(g_pos[0]["gen"])
                pairs["rejected"].append(g_rev_neg[0]["gen"])
                pairs["pair_type"].append("preamble")
                pairs["test_id"].append(test_id)
                pairs["tag"].append(g_pos[0]["tag"])
                pairs["chosen_id"].append(g_pos[0]["chosen"])
                stats["preamble"] += 1

            elif g_pos and guide_rev_split[test_id]["unknown"]:
                pairs["conversations"].append([{"from": "human", "value": g_pos[0]["prompt"]}])
                pairs["chosen"].append(g_pos[0]["gen"])
                pairs["rejected"].append(guide_rev_split[test_id]["unknown"][0]["gen"])
                pairs["pair_type"].append("preamble2unknown")
                pairs["test_id"].append(test_id)
                pairs["tag"].append(g_pos[0]["tag"])
                pairs["chosen_id"].append(g_pos[0]["chosen"])
                stats["preamble2unknown"] += 1
        else:
            stats["num_lost"] += 1

    print("\n=== Pairing Summary ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    return pairs


def domain_split(dataset, tag_field="tag"):
    domains = set(dataset[tag_field])
    splits = {}
    print(f"\nDomains detected: {domains}")

    for tag in domains:
        sub = dataset.filter(lambda x: x[tag_field] == tag)
        splits[tag] = sub.train_test_split(test_size=0.1, seed=42)

    return splits
