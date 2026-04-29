"""Pass 2: trace-based evaluation — latency, tool calls, replayability.

Can be run standalone to add trace metrics to an existing metrics.jsonl,
or called from eval_outputs for a combined single-pass evaluation.
"""

import argparse
import json
import os

from ..mep.writer import iter_meps


# Fields that must be present and non-empty for full replayability
_REPLAY_CHECKS = [
    ("plan", "parsed", "steps"),
    ("plan", "prompt"),
    ("sql_generator", "sql"),           # the actual SQL — most critical for audit
    ("sql_generator", "source_tables"), # citation requirement
    ("sql_generator", "source_fields"), # citation requirement
    ("sql_generator", "tool_trace"),
    ("timestamps", "start"),
    ("timestamps", "end"),
    ("sample", "expected_output"),
    ("sample", "question"),
]


def _get_nested(obj: dict, *keys) -> object:
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def check_replayability(mep: dict) -> float:
    """Return a 0.0–1.0 replayability score.

    1.0 means every required field is present and non-empty.
    """
    present = 0
    for path in _REPLAY_CHECKS:
        val = _get_nested(mep, *path)
        if val not in (None, "", [], {}):
            present += 1
    return present / len(_REPLAY_CHECKS)


def evaluate_trace(mep: dict) -> dict:
    timestamps = mep.get("timestamps", {})
    sql = mep.get("sql_generator", {})
    plan = mep.get("plan", {})
    schema = mep.get("schema_retriever", {})
    sample = mep.get("sample", {})
    config = mep.get("config", {})

    planner_ms    = timestamps.get("planner_ms") or 0
    schema_ms     = timestamps.get("schema_retriever_ms") or 0
    sql_ms        = timestamps.get("sql_generator_ms") or 0
    verifier_ms   = timestamps.get("verifier_ms") or 0

    return {
        "sample_id":               sample.get("sample_id", ""),
        "question_type":           sample.get("question_type", ""),
        "config_name":             config.get("config_name", ""),
        # Latency
        "latency_sec":             (planner_ms + schema_ms + sql_ms + verifier_ms) / 1000.0,
        "planner_latency_sec":     planner_ms / 1000.0,
        "schema_latency_sec":      schema_ms / 1000.0,
        "sql_latency_sec":         sql_ms / 1000.0,
        "verifier_latency_sec":    verifier_ms / 1000.0,
        # SQL-specific
        "sql_tool_call_count":     len(sql.get("tool_trace", [])),
        "guardrail_triggered":     sql.get("guardrail_triggered", False),
        "fallback_used":           sql.get("fallback_used", False),
        "source_tables_count":     len(sql.get("source_tables", [])),
        "citation_present":        len(sql.get("source_tables", [])) > 0,  # key trust signal
        # Parse health
        "planner_parse_ok":        not plan.get("parse_error", True),
        "sql_parse_ok":            not sql.get("parse_error", True),
        "schema_parse_ok":         not schema.get("parse_error", True),
        # Replayability
        "replayability":           check_replayability(mep),
        "error_count":             len(mep.get("errors", [])),
    }


def main() -> None:
    """Compute trace-level metrics for all MEPs and write to JSONL."""
    parser = argparse.ArgumentParser(description="Trace-based MEP evaluation")
    parser.add_argument("--mep_dir", required=True)
    parser.add_argument("--out", default="trace_metrics.jsonl")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f_out:
        count = 0
        for mep in iter_meps(args.mep_dir):
            try:
                metrics = evaluate_trace(mep)
                f_out.write(json.dumps(metrics) + "\n")
                count += 1
            except Exception as exc:
                print(f"Error: {exc}")

    print(f"Done. {count} trace metrics written to {args.out}")


if __name__ == "__main__":
    main()
