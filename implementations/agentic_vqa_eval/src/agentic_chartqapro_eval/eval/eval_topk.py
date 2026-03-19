r"""Top-K answer evaluation pass.

For each MEP, re-queries the VLM asking for the top-3 most likely candidate
answers. Computes hit@1, hit@2, hit@3 without modifying any existing MEPs or
metrics.

Usage:
    uv run --env-file .env -m agentic_chartqapro_eval.eval.eval_topk \
        --mep_dir meps/openai_openai/chartqapro/test \
        --out topk_metrics.jsonl \
        --backend openai \
        --model gpt-4o \
        --k 3
"""

import argparse
import base64
import contextlib
import json
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from openai import OpenAI

from ..langfuse_integration.client import get_client
from ..mep.writer import iter_meps
from ..utils.json_strict import parse_strict
from .eval_outputs import score_answer_accuracy


load_dotenv()

_TOPK_PROMPT = """\
You are a chart reading expert. Analyze the chart image and produce the {k} most \
likely answers to the question, ranked from most to least likely.

Question: {question}
{choices_block}

Inspection plan that was used:
{plan_steps}

Output ONLY JSON, no markdown:
{{"candidates": ["<best answer>", "<2nd best>", "<3rd best>"]}}

Rules:
- Each candidate must be a direct, concise answer (not an explanation)
- If the question is unanswerable, include "UNANSWERABLE" as a candidate
- For MCQ: candidates must come from the provided choices
- JSON only, no extra text
"""


def _encode_image(image_path: str) -> tuple:
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = {
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "png": "png",
        "gif": "gif",
        "webp": "webp",
    }.get(ext, "jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime


def _call_openai_topk(
    image_path: str,
    prompt: str,
    model: str,
    api_key: Optional[str],
) -> str:
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
    b64, mime = _encode_image(image_path)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        max_completion_tokens=256,
        temperature=0.3,  # slight temperature so candidates differ
    )
    return resp.choices[0].message.content or ""


def _call_gemini_topk(
    image_path: str,
    prompt: str,
    model: str,
    api_key: Optional[str],
) -> str:
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
    data, mime = _encode_image(image_path)
    resp = client.models.generate_content(
        model=model,
        contents=[genai.types.Part.from_bytes(data=base64.b64decode(data), mime_type=f"image/{mime}"), prompt],
        config=genai.types.GenerateContentConfig(temperature=0.3, max_output_tokens=256),
    )
    return resp.text or ""


def get_topk_candidates(
    mep: dict,
    k: int = 3,
    backend: str = "gemini",
    model: str = "gemini-2.5-flash-lite",
    api_key: Optional[str] = None,
) -> List[str]:
    """Call the VLM and return up to k candidate answers."""
    sample = mep.get("sample", {})
    plan = mep.get("plan", {}).get("parsed", {})

    image_path = sample.get("image_ref", {}).get("path", "")
    question = sample.get("question", "")
    choices = sample.get("metadata", {}).get("choices")  # may not exist
    plan_steps = plan.get("steps", [])

    choices_block = f"Choices: {', '.join(choices)}" if choices else ""
    steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan_steps)) or "  (none)"

    prompt = _TOPK_PROMPT.format(
        k=k,
        question=question,
        choices_block=choices_block,
        plan_steps=steps_text,
    )

    if not image_path or not Path(image_path).exists():
        return []

    try:
        if backend == "openai":
            raw = _call_openai_topk(image_path, prompt, model, api_key)
        else:
            raw = _call_gemini_topk(image_path, prompt, model, api_key)

        parsed, _ = parse_strict(raw, required_keys=["candidates"])
        candidates = parsed.get("candidates", [])
        return [str(c).strip() for c in candidates[:k] if c]
    except Exception as exc:
        print(f"  topk error for {sample.get('sample_id', '?')}: {exc}")
        return []


def _hit_at_k(expected: str, candidates: List[str], question_type: str, k: int) -> float:
    """1.0 if expected matches any of the first k candidates."""
    for c in candidates[:k]:
        if score_answer_accuracy(expected, c, question_type) > 0:
            return 1.0
    return 0.0


def evaluate_topk(
    mep: dict,
    k: int = 3,
    backend: str = "gemini",
    model: str = "gemini-2.5-flash-lite",
    api_key: Optional[str] = None,
) -> dict:
    """Evaluate top-K answer candidates for a single MEP."""
    sample = mep.get("sample", {})
    config = mep.get("config", {})

    expected = sample.get("expected_output", "")
    question_type = sample.get("question_type", "standard")

    # Top-1 answer from the original MEP (already computed, no extra call)
    original_answer = mep.get("vision", {}).get("parsed", {}).get("answer", "")

    candidates = get_topk_candidates(mep, k=k, backend=backend, model=model, api_key=api_key)

    result: dict = {
        "sample_id": sample.get("sample_id", ""),
        "question_type": question_type,
        "config_name": config.get("config_name", ""),
        "expected": expected,
        "original_answer": original_answer,
        "topk_candidates": candidates,
        "original_accuracy": score_answer_accuracy(expected, original_answer, question_type),
    }

    for ki in range(1, k + 1):
        result[f"hit_at_{ki}"] = _hit_at_k(expected, candidates, question_type, ki)

    return result


def main() -> None:
    """Run top-K evaluation on MEPs and write results to JSONL."""
    parser = argparse.ArgumentParser(description="Top-K answer candidate evaluation")
    parser.add_argument("--mep_dir", required=True)
    parser.add_argument("--out", default="topk_metrics.jsonl")
    parser.add_argument("--backend", default="gemini", choices=["openai", "gemini"])
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=None, help="Limit to first N MEPs")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "") if args.backend == "openai" else os.environ.get("GEMINI_API_KEY", "")

    lf_client = get_client()

    with open(args.out, "w") as f_out:
        count = 0
        for mep in iter_meps(args.mep_dir):
            if args.n is not None and count >= args.n:
                break
            try:
                result = evaluate_topk(
                    mep,
                    k=args.k,
                    backend=args.backend,
                    model=args.model,
                    api_key=api_key,
                )
                f_out.write(json.dumps(result) + "\n")
                sid = result["sample_id"]
                exp = result["expected"]
                cands = result["topk_candidates"]
                h1 = result.get("hit_at_1", 0)
                h3 = result.get(f"hit_at_{args.k}", 0)
                print(f"  {sid}  exp={exp!r}  candidates={cands}  hit@1={h1}  hit@{args.k}={h3}")

                lf_trace_id = mep.get("lf_trace_id")
                if lf_client and lf_trace_id:
                    for ki in range(1, args.k + 1):
                        key = f"hit_at_{ki}"
                        if key in result:
                            with contextlib.suppress(Exception):
                                lf_client.create_score(
                                    trace_id=lf_trace_id,
                                    name=key,
                                    value=float(result[key]),
                                )

                count += 1
            except Exception as exc:
                print(f"  Error: {exc}")

    # Print summary
    with open(args.out) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if records:
        print(f"\n--- Top-K Summary (n={len(records)}) ---")
        orig_acc = sum(r["original_accuracy"] for r in records) / len(records)
        print(f"  Original accuracy (hit@1 from MEP) : {orig_acc:.3f}")
        for ki in range(1, args.k + 1):
            key = f"hit_at_{ki}"
            if key in records[0]:
                h = sum(r[key] for r in records) / len(records)
                print(f"  hit@{ki}                              : {h:.3f}")
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
