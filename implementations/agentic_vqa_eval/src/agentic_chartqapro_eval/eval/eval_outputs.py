r"""Pass 1: output-based evaluation — rule-based accuracy + LLM judge rubric.

Usage:
    uv run --env-file .env -m agentic_chartqapro_eval.eval.eval_outputs \\
        --mep_dir meps/openai_gemini/chartqapro/test \\
        --out metrics.jsonl
"""

import argparse
import contextlib
import json
import math
import os
import re
from typing import Optional

from dotenv import load_dotenv

from ..langfuse_integration.client import get_client
from ..mep.writer import iter_meps
from .judge import judge_mep


load_dotenv()


# ---------------------------------------------------------------------------
# Rule-based scorers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s\-\.]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _to_number(text: str) -> Optional[float]:
    text = text.replace(",", "").replace("%", "").strip()
    m = re.search(r"-?\d+\.?\d*", text)
    try:
        return float(m.group()) if m else None
    except ValueError:
        return None


def score_answer_accuracy(expected: str, predicted: str, question_type: str) -> float:
    """Exact-match with numeric tolerance and MCQ partial credit."""
    exp = _normalize(expected)
    pred = _normalize(predicted)

    if exp == pred:
        return 1.0

    exp_num = _to_number(exp)
    pred_num = _to_number(pred)

    if exp_num is not None and pred_num is not None:
        # Direct numeric match (rel_tol=0.1%, abs_tol=0.05 for near-zero values)
        if math.isclose(exp_num, pred_num, rel_tol=0.001, abs_tol=0.05):
            return 1.0

        # Decimal/percentage normalization: agent returned 0.2212, expected 22.12
        # Only applies when one value is in (0, 1] and the other is in (0, 100].
        if 0 < pred_num <= 1 and 0 < exp_num <= 100:
            if math.isclose(pred_num * 100, exp_num, rel_tol=0.001, abs_tol=0.5):
                return 1.0
        if 0 < exp_num <= 1 and 0 < pred_num <= 100:
            if math.isclose(exp_num * 100, pred_num, rel_tol=0.001, abs_tol=0.5):
                return 1.0

    # MCQ substring check
    if question_type == "mcq" and (exp in pred or pred in exp):
        return 0.5

    return 0.0


def score_unanswerable(expected: str, predicted: str) -> Optional[float]:
    """Score binary classification for unanswerable samples.

    Returns None if expected is NOT UNANSWERABLE (metric not applicable).
    """
    exp_ua = expected.strip().upper() == "UNANSWERABLE"
    pred_ua = predicted.strip().upper() == "UNANSWERABLE"
    if exp_ua:
        return 1.0 if pred_ua else 0.0
    return None


# ---------------------------------------------------------------------------
# Per-MEP evaluation
# ---------------------------------------------------------------------------


def evaluate_mep(
    mep: dict,
    use_judge: bool = True,
    judge_backend: str = "anthropic",     # switch default — you're on Claude
    judge_model: str = "claude-sonnet-4-6",
) -> dict:
    sample     = mep.get("sample", {})
    plan       = mep.get("plan", {})
    sql        = mep.get("sql_generator", {})    # was vision
    verifier   = mep.get("verifier") or {}
    timestamps = mep.get("timestamps", {})
    config     = mep.get("config", {})

    expected       = sample.get("expected_output", "")
    sql_parsed     = sql.get("parsed", {})
    verifier_parsed = verifier.get("parsed") or {}
    verifier_verdict = verifier.get("verdict", "skipped")

    # Final answer: prefer verifier output, fall back to sql_generator
    predicted    = verifier_parsed.get("answer") or sql_parsed.get("answer", "")
    question_type = sample.get("question_type", "standard")

    planner_ms  = timestamps.get("planner_ms") or 0
    sql_ms      = timestamps.get("sql_generator_ms") or 0
    verifier_ms = timestamps.get("verifier_ms") or 0

    metrics: dict = {
        "sample_id":          sample.get("sample_id", ""),
        "question_type":      question_type,
        "config_name":        config.get("config_name", ""),
        "expected":           expected,
        "predicted":          predicted,
        "sql_answer":         sql_parsed.get("answer", ""),   # raw pre-verification
        "sql_query":          sql.get("sql", ""),             # the actual SQL — new
        "source_tables":      sql.get("source_tables", []),   # for citation audit
        "source_fields":      sql.get("source_fields", []),   # for citation audit
        "verifier_verdict":   verifier_verdict,
        "planner_parse_ok":   not plan.get("parse_error", True),
        "sql_parse_ok":       not sql.get("parse_error", True),
        "json_parse_ok":      (not plan.get("parse_error", True))
                              and (not sql.get("parse_error", True)),
        "citation_present":   len(sql.get("source_tables", [])) > 0,
        "guardrail_triggered": sql.get("guardrail_triggered", False),
        "answer_accuracy":    score_answer_accuracy(expected, predicted, question_type),
        "latency_sec":        (planner_ms + sql_ms + verifier_ms) / 1000.0,
        "tool_call_count":    len(sql.get("tool_trace", [])),
        "has_errors":         len(mep.get("errors", [])) > 0,
    }

    ua = score_unanswerable(expected, predicted)
    if ua is not None:
        metrics["unanswerable_accuracy"] = ua

    if use_judge:
        judge_scores = judge_mep(mep, backend=judge_backend, model=judge_model)
        for k, v in judge_scores.items():
            metrics[f"judge_{k}"] = v

    # Langfuse trace logging — unchanged
    lf_trace_id = mep.get("lf_trace_id")
    if lf_trace_id:
        client = get_client()
        if client:
            score_keys = ["answer_accuracy", "citation_present", "latency_sec"] + (
                [f"judge_{k}" for k in judge_scores] if use_judge else []
            )
            scores = {k: metrics[k] for k in score_keys if isinstance(metrics.get(k), (int, float, bool))}
            for k, v in scores.items():
                with contextlib.suppress(Exception):
                    client.create_score(trace_id=lf_trace_id, name=k, value=float(v))

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Evaluate MEPs and write output-based metrics to JSONL."""
    parser = argparse.ArgumentParser(description="Evaluate MEPs — output-based metrics")
    parser.add_argument("--mep_dir", required=True, help="Directory containing MEP JSON files")
    parser.add_argument("--out", default="metrics.jsonl", help="Output JSONL file")
    parser.add_argument("--no_judge", action="store_true", help="Skip LLM judge (faster)")
    parser.add_argument("--judge_backend", default="gemini", choices=["openai", "gemini"])
    parser.add_argument("--judge_model", default="gemini-2.5-flash-lite")
    args = parser.parse_args()

    use_judge = not args.no_judge

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f_out:
        count = 0
        for mep in iter_meps(args.mep_dir):
            try:
                metrics = evaluate_mep(
                    mep,
                    use_judge=use_judge,
                    judge_backend=args.judge_backend,
                    judge_model=args.judge_model,
                )
                f_out.write(json.dumps(metrics) + "\n")
                count += 1
                if count % 10 == 0:
                    print(f"  evaluated {count} samples …")
            except Exception as exc:
                print(f"  Error evaluating MEP: {exc}")

    print(f"Done. {count} metrics written to {args.out}")


if __name__ == "__main__":
    main()
