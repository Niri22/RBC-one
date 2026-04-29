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
You are a SQL-based metrics assistant. A previous attempt to answer this question \
may have been incorrect. Generate a revised SQL query and answer.

Question: {question}
Known source tables: {source_tables}
Attempt: {attempt} of {k}

Previous SQL (may be wrong):
{previous_sql}

Plan steps used:
{plan_steps}

Output ONLY JSON, no markdown:
{{"answer": "<concise metric answer>", "sql": "<revised SQL query>"}}

Rules:
- Answer must be a direct, concise metric value
- SQL must reference only the known source tables
- If the question is unanswerable with available data, answer "UNANSWERABLE"
"""
def _call_llm(
    prompt: str,
    backend: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
) -> str:
    """Call a text LLM for top-K SQL re-querying."""
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=256,
        )
        return resp.choices[0].message.content or ""

    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""))
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text or ""

    if backend == "gemini":
        from google import genai
        client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.text or ""

    raise ValueError(f"Unknown topk backend: {backend!r}")




def get_topk_candidates(
    mep: dict,
    k: int = 3,
    backend: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
    ) -> List[str]:
    """Re-query the SQL generator up to k times with revised prompts.
    Returns list of candidate answers."""
    sample = mep.get("sample", {})
    plan   = mep.get("plan", {}).get("parsed", {})
    sql    = mep.get("sql_generator", {})

    question    = sample.get("question", "")
    original_sql = sql.get("sql", "")
    plan_steps  = plan.get("steps", [])
    source_tables = sql.get("source_tables", [])

    candidates = []
    previous_sql = original_sql

    for attempt in range(k):
        prompt = _TOPK_PROMPT.format(
            k=k,
            question=question,
            plan_steps="\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan_steps)),
            previous_sql=previous_sql,
            attempt=attempt + 1,
            source_tables=", ".join(source_tables) or "unknown",
        )
        try:
            raw = _call_llm(prompt, backend=backend, model=model, api_key=api_key)
            parsed, _ = parse_strict(raw, required_keys=["answer", "sql"])
            answer = str(parsed.get("answer", "")).strip()
            previous_sql = parsed.get("sql", previous_sql)  # use revised SQL next round
            if answer:
                candidates.append(answer)
        except Exception as exc:
            print(f"  topk attempt {attempt+1} error: {exc}")

    return candidates[:k]


def _hit_at_k(expected: str, candidates: List[str], question_type: str, k: int) -> float:
    """1.0 if expected matches any of the first k candidates."""
    for c in candidates[:k]:
        if score_answer_accuracy(expected, c, question_type) > 0:
            return 1.0
    return 0.0


def evaluate_topk(
    mep: dict,
    k: int = 3,
    backend: str = "anthropic",          # add back the missing params
    model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
    ) -> dict: 
    sample   = mep.get("sample", {})
    config   = mep.get("config", {})
    sql      = mep.get("sql_generator", {})      # was vision

    expected      = sample.get("expected_output", "")
    question_type = sample.get("question_type", "standard")
    original_answer = (mep.get("verifier", {}) or {}).get("parsed", {}).get("answer") \
                      or sql.get("parsed", {}).get("answer", "")   # prefer verified answer

    candidates = get_topk_candidates(mep, k=k, backend=backend, model=model, api_key=api_key)

    result = {
        "sample_id":         sample.get("sample_id", ""),
        "question_type":     question_type,
        "config_name":       config.get("config_name", ""),
        "expected":          expected,
        "original_answer":   original_answer,
        "topk_candidates":   candidates,
        "original_accuracy": score_answer_accuracy(expected, original_answer, question_type),
        "citation_present":  len(sql.get("source_tables", [])) > 0,  # carry forward
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
