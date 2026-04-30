r"""Pass 4: failure taxonomy — classify WHY each wrong SQL answer failed.

For each MEP where the answer is incorrect, calls a text LLM with the question,
expected value, predicted value, SQL query, and plan to classify the primary
failure mode into one of the SQL-specific categories below.

No chart images are involved — classification is grounded in SQL evidence only.

Usage:
    uv run --env-file .env -m agentic_chartqapro_eval.eval.error_taxonomy \
        --mep_dir meps/openai_openai/sql/test \
        --metrics_file metrics.jsonl \
        --out taxonomy.jsonl \
        --backend anthropic \
        --model claude-sonnet-4-6
"""

import argparse
import contextlib
import json
import os
from typing import Optional

from dotenv import load_dotenv

from ..langfuse_integration.client import get_client
from ..mep.writer import iter_meps
from ..utils.json_strict import parse_strict


load_dotenv()

# ---------------------------------------------------------------------------
# SQL-specific taxonomy categories
# ---------------------------------------------------------------------------

TAXONOMY_CATEGORIES = [
    "wrong_table",              # queried the wrong table or dataset
    "wrong_aggregation",        # wrong aggregate function (SUM vs COUNT, AVG vs SUM, etc.)
    "wrong_filter",             # WHERE clause incorrect, missing, or over-inclusive
    "date_range_error",         # wrong time window, date column, or period boundary
    "join_error",               # incorrect JOIN logic, missing JOIN, or wrong join key
    "metric_definition_mismatch",  # correct SQL structure but misunderstood the KPI definition
    "guardrail_blocked",        # query was blocked by guardrails (SELECT *, unknown table, etc.)
    "parse_failure",            # LLM output could not be parsed as valid SQL or JSON
    "unanswerable_failure",     # should say UNANSWERABLE but didn't, or vice versa
    "question_misunderstanding",  # answered a different or adjacent question
    "other",                    # none of the above
]

_TAXONOMY_PROMPT = """\
You are a senior data engineer reviewing a failed SQL metrics query.

--- CONTEXT ---
Question      : {question}
Expected value: {expected}
Predicted value: {predicted}   ← WRONG
Verifier verdict: {verifier_verdict}

--- SQL EVIDENCE ---
SQL query used:
{sql}

Source tables cited: {source_tables}
Source fields cited: {source_fields}
Guardrail triggered: {guardrail_triggered}

--- PLAN STEPS THAT SHOULD HAVE BEEN FOLLOWED ---
{plan_steps}

--- TASK ---
Classify the PRIMARY reason this answer is wrong into exactly ONE category:

  wrong_table              – queried the wrong table or dataset
  wrong_aggregation        – wrong aggregate (SUM vs COUNT, AVG vs SUM, etc.)
  wrong_filter             – WHERE clause incorrect, missing, or over-inclusive
  date_range_error         – wrong time window, date column, or period boundary
  join_error               – incorrect JOIN logic, missing JOIN, or wrong join key
  metric_definition_mismatch – correct SQL structure but misunderstood the KPI
  guardrail_blocked        – query was blocked by guardrails (SELECT *, unknown table)
  parse_failure            – output could not be parsed as valid SQL or JSON
  unanswerable_failure     – should say UNANSWERABLE but didn't, or vice versa
  question_misunderstanding – answered a different or adjacent question
  other                    – none of the above

Output ONLY valid JSON, no markdown:
{{"failure_type": "<category>", "failure_reason": "<one sentence citing the SQL evidence>"}}
"""

_TAXONOMY_KEYS = ["failure_type", "failure_reason"]


# ---------------------------------------------------------------------------
# LLM helpers (text-only — no images)
# ---------------------------------------------------------------------------


def _call_llm(prompt: str, backend: str, model: str, api_key: Optional[str]) -> str:
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=256,
        )
        return resp.choices[0].message.content or ""

    if backend == "gemini":
        from google import genai
        client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.text or ""

    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""))
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text or ""

    raise ValueError(f"Unknown taxonomy backend: {backend!r}")


# ---------------------------------------------------------------------------
# Per-MEP classification
# ---------------------------------------------------------------------------


def classify_failure(
    mep: dict,
    answer_accuracy: float,
    backend: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
) -> dict:
    """Classify WHY the SQL agent failed on this sample.

    Returns a dict with ``failure_type`` and ``failure_reason``.
    If ``answer_accuracy >= 1.0``, returns immediately with
    ``failure_type="correct"`` without making an LLM call.

    Guardrail-blocked samples are classified immediately without an LLM call.
    """
    if answer_accuracy >= 1.0:
        return {"failure_type": "correct", "failure_reason": ""}

    sample = mep.get("sample", {})
    plan = mep.get("plan", {}).get("parsed", {})
    sql = mep.get("sql_generator", {})
    verifier = mep.get("verifier") or {}

    question = sample.get("question", "")
    expected = sample.get("expected_output", "")
    predicted = (
        (verifier.get("parsed") or {}).get("answer")
        or sql.get("parsed", {}).get("answer", "")
    )
    verifier_verdict = verifier.get("verdict", "skipped")
    sql_query = sql.get("sql", "") or sql.get("parsed", {}).get("sql", "")
    source_tables = ", ".join(sql.get("source_tables", [])) or "none cited"
    source_fields = ", ".join(sql.get("source_fields", [])) or "none cited"
    guardrail_triggered = sql.get("guardrail_triggered", False)
    plan_steps = plan.get("steps", [])
    steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan_steps)) or "  (none)"

    # Short-circuit for guardrail blocks — no LLM call needed
    if guardrail_triggered:
        return {
            "failure_type": "guardrail_blocked",
            "failure_reason": "Query was blocked by guardrail before execution.",
        }

    # Short-circuit for parse failures — SQL string is empty and no answer
    if not sql_query and not predicted:
        return {
            "failure_type": "parse_failure",
            "failure_reason": "SQL generator produced no parseable output.",
        }

    prompt = _TAXONOMY_PROMPT.format(
        question=question or "(unknown)",
        expected=expected,
        predicted=predicted or "(none)",
        verifier_verdict=verifier_verdict,
        sql=sql_query or "(none)",
        source_tables=source_tables,
        source_fields=source_fields,
        guardrail_triggered=guardrail_triggered,
        plan_steps=steps_text,
    )

    try:
        raw = _call_llm(prompt, backend, model, api_key)
        result, ok = parse_strict(raw, required_keys=_TAXONOMY_KEYS)
        if not result:
            return {
                "failure_type": "other",
                "failure_reason": raw[:200],
                "parse_error": True,
            }

        ft = result.get("failure_type", "other").strip().lower()
        if ft not in TAXONOMY_CATEGORIES:
            ft = "other"
        result["failure_type"] = ft
        return result

    except Exception as exc:
        return {"failure_type": "other", "failure_reason": f"Taxonomy error: {exc}"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Classify SQL failures in MEP files and write taxonomy results to JSONL."""
    parser = argparse.ArgumentParser(description="Pass 4: SQL failure taxonomy")
    parser.add_argument("--mep_dir", required=True, help="Directory containing MEP JSON files")
    parser.add_argument(
        "--metrics_file",
        default=None,
        help="Optional metrics.jsonl from eval_outputs — used to look up answer_accuracy",
    )
    parser.add_argument("--out", default="taxonomy.jsonl", help="Output JSONL file")
    parser.add_argument(
        "--backend", default="anthropic", choices=["openai", "gemini", "anthropic"]
    )
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument(
        "--all",
        dest="classify_all",
        action="store_true",
        help="Classify correct samples too (by default they are skipped)",
    )
    parser.add_argument("--n", type=int, default=None, help="Limit to first N MEPs")
    args = parser.parse_args()

    api_key_env = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    api_key = os.environ.get(api_key_env[args.backend], "")

    # Build accuracy lookup from metrics.jsonl if provided
    accuracy_by_id: dict = {}
    if args.metrics_file:
        from pathlib import Path
        if Path(args.metrics_file).exists():
            with open(args.metrics_file) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        accuracy_by_id[row.get("sample_id", "")] = row.get("answer_accuracy", 0.0)

    lf_client = get_client()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w") as f_out:
        count = 0
        skipped = 0
        for mep in iter_meps(args.mep_dir):
            if args.n is not None and count >= args.n:
                break

            sample = mep.get("sample", {})
            sql = mep.get("sql_generator", {})
            verifier = mep.get("verifier") or {}

            sample_id = sample.get("sample_id", "")
            config_name = mep.get("config", {}).get("config_name", "")
            question_type = sample.get("question_type", "standard")
            expected = sample.get("expected_output", "")
            predicted = (
                (verifier.get("parsed") or {}).get("answer")
                or sql.get("parsed", {}).get("answer", "")
            )

            answer_accuracy = accuracy_by_id.get(sample_id, -1.0)
            if answer_accuracy < 0:
                answer_accuracy = 1.0 if expected.strip().lower() == predicted.strip().lower() else 0.0

            if answer_accuracy >= 1.0 and not args.classify_all:
                skipped += 1
                f_out.write(json.dumps({
                    "sample_id": sample_id,
                    "config_name": config_name,
                    "question_type": question_type,
                    "expected": expected,
                    "predicted": predicted,
                    "answer_accuracy": answer_accuracy,
                    "failure_type": "correct",
                    "failure_reason": "",
                }) + "\n")
                count += 1
                continue

            try:
                result = classify_failure(
                    mep, answer_accuracy, backend=args.backend, model=args.model, api_key=api_key
                )
                row = {
                    "sample_id": sample_id,
                    "config_name": config_name,
                    "question_type": question_type,
                    "expected": expected,
                    "predicted": predicted,
                    "answer_accuracy": answer_accuracy,
                    **result,
                }
                f_out.write(json.dumps(row) + "\n")
                count += 1

                lf_trace_id = mep.get("lf_trace_id")
                if lf_client and lf_trace_id:
                    with contextlib.suppress(Exception):
                        lf_client.create_score(
                            trace_id=lf_trace_id,
                            name=f"failure_{result.get('failure_type', 'other')}",
                            value=1.0,
                        )

                if count % 10 == 0:
                    print(f"  classified {count} samples …")

            except Exception as exc:
                print(f"  Error on {sample_id}: {exc}")

    print(f"\nDone. {count} rows written to {args.out}  ({skipped} correct samples recorded as-is)")

    breakdown: dict = {}
    with open(args.out) as f:
        for line in f:
            ft = json.loads(line).get("failure_type", "?")
            breakdown[ft] = breakdown.get(ft, 0) + 1
    print("\nFailure type breakdown:")
    for ft, n in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"  {ft:<35} {n}")


if __name__ == "__main__":
    main()
