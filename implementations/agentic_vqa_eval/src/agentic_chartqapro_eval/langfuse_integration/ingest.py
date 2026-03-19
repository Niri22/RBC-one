"""Retroactive ingestion: convert existing MEP JSON files to Langfuse Traces.

This lets you visualise runs that completed before Langfuse was wired in.

Usage:
    python -m agentic_chartqapro_eval.langfuse_integration.ingest \
        --mep_dir meps/openai_openai/chartqapro/test \
        [--metrics_file metrics.jsonl]
"""

import argparse
import contextlib
import json
from pathlib import Path
from typing import Optional

from .client import get_client
from .tracing import _normalize_usage


def ingest_mep(
    mep: dict,
    client: object,
    metrics: Optional[dict] = None,
    project_name: str = "chartqapro-eval",  # noqa: ARG001 — kept for API compat
) -> None:
    """Create a Langfuse Trace from a single MEP dict (retroactively)."""
    sample = mep.get("sample", {})
    plan = mep.get("plan", {})
    vision = mep.get("vision", {})
    config = mep.get("config", {})

    sample_id = sample.get("sample_id", "unknown")
    config_name = config.get("config_name", "unknown")
    question_type = sample.get("question_type", "standard")
    question = sample.get("question", "")
    expected = sample.get("expected_output", "")
    vision_parsed = vision.get("parsed", {})

    with client.start_as_current_observation(  # type: ignore[union-attr]
        name=f"chartqapro/{sample_id}",
        as_type="span",
        input={"question": question, "expected_output": expected},
        output=vision_parsed if vision_parsed else None,
        metadata={
            "run_id": mep.get("run_id", ""),
            "config": config_name,
            "question_type": question_type,
            "schema_version": mep.get("schema_version", ""),
            "has_errors": bool(mep.get("errors")),
            "retroactive": True,
        },
    ) as trace_span:
        # Planner generation
        if plan.get("prompt"):
            planner_gen = trace_span.start_observation(
                name="planner",
                as_type="generation",
                input={"prompt": plan.get("prompt", "")},
                model=config.get("planner_model", ""),
                metadata={"backend": config.get("planner_backend", "")},
            )
            planner_gen.update(
                output={
                    "plan": plan.get("parsed", {}),
                    "parse_error": plan.get("parse_error", False),
                }
            )
            planner_gen.end()

        # Vision tool generations — one per ToolTrace entry
        for tt in vision.get("tool_trace", []):
            usage = tt.get("provider_metadata", {}).get("usage", {})
            tool_gen = trace_span.start_observation(
                name="vision_qa_tool",
                as_type="generation",
                input={
                    "question": question,
                    "plan_steps": plan.get("parsed", {}).get("steps", []),
                },
                model=tt.get("model", config.get("vision_model", "")),
                metadata={
                    "backend": tt.get("backend", config.get("vision_backend", "")),
                    "elapsed_ms": tt.get("elapsed_ms"),
                },
                usage_details=_normalize_usage(usage) if usage else None,
            )
            tool_gen.update(output=vision_parsed if vision_parsed else None)
            tool_gen.end()

        # Attach evaluation scores if provided
        if metrics:
            for key in [
                "answer_accuracy",
                "judge_explanation_quality",
                "judge_hallucination_rate",
                "judge_plan_coverage",
                "judge_plan_adherence",
                "judge_faithfulness_alignment",
            ]:
                if key in metrics and isinstance(metrics[key], (int, float)):
                    with contextlib.suppress(Exception):
                        trace_span.score_trace(name=key, value=float(metrics[key]))


def ingest_dir(
    mep_dir: str,
    metrics_file: Optional[str] = None,
    project_name: str = "chartqapro-eval",
) -> int:
    """Ingest all MEPs from a directory. Returns the number ingested."""
    client = get_client()
    if client is None:
        print("[langfuse] No client — set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        return 0

    metrics_by_id: dict = {}
    if metrics_file and Path(metrics_file).exists():
        with open(metrics_file) as f:
            for raw_line in f:
                line = raw_line.strip()
                if line:
                    row = json.loads(line)
                    metrics_by_id[row.get("sample_id", "")] = row

    mep_path = Path(mep_dir)
    mep_files = list(mep_path.glob("*.json"))
    if not mep_files:
        print(f"[langfuse] No MEP JSON files found in {mep_dir}")
        return 0

    count = 0
    for fpath in sorted(mep_files):
        try:
            mep = json.loads(fpath.read_text())
            sample_id = mep.get("sample", {}).get("sample_id", "")
            ingest_mep(
                mep,
                client,
                metrics=metrics_by_id.get(sample_id),
                project_name=project_name,
            )
            count += 1
            print(f"  ingested {sample_id}")
        except Exception as exc:
            print(f"  ERROR {fpath.name}: {exc}")

    print(f"[langfuse] Ingested {count}/{len(mep_files)} MEPs from {mep_dir}")
    with contextlib.suppress(Exception):
        client.flush()  # type: ignore[union-attr]
    return count


def main() -> None:
    """Parse CLI arguments and ingest MEP files into Langfuse."""
    parser = argparse.ArgumentParser(description="Ingest existing MEPs into Langfuse")
    parser.add_argument("--mep_dir", required=True, help="Directory containing MEP JSON files")
    parser.add_argument(
        "--metrics_file",
        default=None,
        help="Optional metrics.jsonl for feedback scores",
    )
    parser.add_argument("--project", default="chartqapro-eval", help="Langfuse project name (metadata)")
    args = parser.parse_args()

    ingest_dir(args.mep_dir, args.metrics_file, project_name=args.project)


if __name__ == "__main__":
    main()
