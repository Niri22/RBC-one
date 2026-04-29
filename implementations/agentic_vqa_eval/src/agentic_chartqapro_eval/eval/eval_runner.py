"""Eval harness runner — loads eval samples, runs full pipeline, writes MEPs.

Initialises all agents once and reuses them across the eval loop.
Stubs out eval_outputs.jsonl, eval_traces.jsonl, eval_topk.jsonl on start
so downstream eval passes can begin writing immediately.

Usage:
    uv run --env-file .env -m rbc_metrics_eval.eval.eval_runner \\
        --samples eval_samples.json \\
        --mep_dir meps/run_001 \\
        --db_uri sqlite:///rbc_metrics.db \\
        --backend anthropic \\
        --model claude-sonnet-4-6
"""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from ..agents.planner_agent import PlannerAgent
from ..agents.sql_generator_agent import SQLGeneratorAgent
from ..agents.schema_retriever import SchemaRetrieverTool
from ..agents.verifier_agent import VerifierAgent
from ..langfuse_integration.client import get_client
from ..mep.schema import (
    MEP, MEPConfig, MEPPlan, MEPSample, MEPSQLGenerator,
    MEPSchemaRetriever, MEPTimestamps, MEPVerifier,
)
from ..mep.writer import write_mep


load_dotenv()

# ── Stub filenames written at startup ─────────────────────────────────────────
_STUB_FILES = {
    "eval_outputs": "eval_outputs.jsonl",
    "eval_traces":  "eval_traces.jsonl",
    "eval_topk":    "eval_topk.jsonl",
}


# ── Sample loader ──────────────────────────────────────────────────────────────

def load_eval_samples(path: str) -> list[dict]:
    """Load eval_samples.json and return list of sample dicts."""
    with open(path) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} eval samples from {path}")
    return samples


def sample_to_mep_sample(s: dict) -> MEPSample:
    """Convert a raw eval sample dict to a typed MEPSample."""
    return MEPSample(
        dataset="credit_card_clients",
        sample_id=s["sample_id"],
        question=s["question"],
        question_type=s["question_type"],
        expected_output=s["expected_output"],
        metadata={
            **s.get("metadata", {}),
            "kpi_name": s.get("kpi_name", ""),
            "expected_sql": s.get("expected_sql"),   # optional — used in eval_traces
        },
    )


# ── MEP assembly helpers ───────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed(start_iso: str, end_iso: str) -> float:
    """Return elapsed milliseconds between two ISO timestamps."""
    fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
    try:
        s = datetime.fromisoformat(start_iso)
        e = datetime.fromisoformat(end_iso)
        return (e - s).total_seconds() * 1000
    except Exception:
        return 0.0


def _make_config(backend: str, model: str, config_name: str) -> MEPConfig:
    return MEPConfig(
        planner_backend=backend,
        sql_backend=backend,
        judge_backend="anthropic",
        config_name=config_name,
        planner_model=model,
        sql_model=model,
        schema_retriever_enabled=True,
        verifier_enabled=True,
    )


# ── Stub writer ────────────────────────────────────────────────────────────────

def _write_stubs(out_dir: str) -> dict[str, str]:
    """
    Create empty stub JSONL files for each eval pass.

    This lets eval_outputs, eval_traces, and eval_topk start appending
    immediately without needing to check if the file exists.
    Returns a dict of {key: filepath}.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, filename in _STUB_FILES.items():
        fpath = os.path.join(out_dir, filename)
        # Write empty file — do not clobber if re-running from checkpoint
        if not os.path.exists(fpath):
            open(fpath, "w").close()
            print(f"  Stubbed: {fpath}")
        paths[key] = fpath
    return paths


# ── Per-sample pipeline ────────────────────────────────────────────────────────

def run_sample(
    sample_dict: dict,
    planner: PlannerAgent,
    schema_retriever: SchemaRetrieverTool,
    sql_generator: SQLGeneratorAgent,
    verifier: VerifierAgent,
    config: MEPConfig,
    lf_client: Any,
    run_id: str,
) -> MEP:
    """
    Run a single eval sample through the full pipeline and return a populated MEP.

    Each agent appends to its own MEP field. Errors at any step are caught,
    logged to mep.errors, and execution continues to the next step where possible.
    """
    mep_sample = sample_to_mep_sample(sample_dict)

    mep = MEP(
        schema_version="mep.v2",
        run_id=run_id,
        config=config,
        sample=mep_sample,
        errors=[],
    )

    lf_trace = None
    if lf_client:
        lf_trace = lf_client.trace(
            name="rbc_metrics_pipeline",
            input={"question": mep_sample.question, "sample_id": mep_sample.sample_id},
            metadata={"run_id": run_id},
        )
        mep.lf_trace_id = lf_trace.id

    run_start = _now_iso()
    planner_ms = schema_ms = sql_ms = verifier_ms = 0.0

    # ── Step 1: PlannerAgent ──────────────────────────────────────────────
    try:
        t0 = _now_iso()
        prompt, parsed_plan, parse_error, raw_plan = planner.run(mep_sample, lf_trace)
        t1 = _now_iso()
        planner_ms = _elapsed(t0, t1)
        mep.plan = MEPPlan(
            prompt=prompt,
            raw_text=raw_plan,
            parsed=parsed_plan,
            parse_error=parse_error,
        )
    except Exception as exc:
        mep.errors.append(f"planner: {exc}")
        parsed_plan = {}

    # ── Step 2: SchemaRetrieverTool (optional) ────────────────────────────
    schema_context = None
    if config.schema_retriever_enabled and parsed_plan:
        try:
            t0 = _now_iso()
            schema_result = schema_retriever.run(parsed_plan)
            t1 = _now_iso()
            schema_ms = _elapsed(t0, t1)
            mep.schema_retriever = schema_result   # MEPSchemaRetriever dataclass
        except Exception as exc:
            mep.errors.append(f"schema_retriever: {exc}")
        else:
            schema_context = mep.schema_retriever

    # ── Step 3: SQLGeneratorAgent ─────────────────────────────────────────
    parsed_sql: dict = {}
    if parsed_plan:
        try:
            t0 = _now_iso()
            sql_prompt, parsed_sql, parse_error, raw_sql = sql_generator.run(
                mep_sample, parsed_plan, schema_context, lf_trace
            )
            t1 = _now_iso()
            sql_ms = _elapsed(t0, t1)
            mep.sql_generator = MEPSQLGenerator(
                prompt=sql_prompt,
                raw_text=raw_sql,
                sql=parsed_sql.get("sql", ""),
                parsed=parsed_sql,
                source_tables=parsed_sql.get("source_tables", []),
                source_fields=parsed_sql.get("source_fields", []),
                data_freshness=parsed_sql.get("data_freshness", ""),
                parse_error=parse_error,
                guardrail_triggered=parsed_sql.get("guardrail_triggered", False),
                fallback_used=parsed_sql.get("fallback_used", False),
            )
            # Hard stop: citation missing → log error, skip verifier
            if not mep.sql_generator.source_tables:
                mep.errors.append("sql_generator: source_tables empty — citation requirement violated")
        except Exception as exc:
            mep.errors.append(f"sql_generator: {exc}")

    # ── Step 4: VerifierAgent (optional) ──────────────────────────────────
    if config.verifier_enabled and parsed_sql and not parsed_sql.get("guardrail_triggered"):
        try:
            t0 = _now_iso()
            v_prompt, v_parsed, v_parse_error, v_raw, verdict = verifier.run(
                mep_sample, parsed_plan, parsed_sql, lf_trace
            )
            t1 = _now_iso()
            verifier_ms = _elapsed(t0, t1)
            mep.verifier = MEPVerifier(
                prompt=v_prompt,
                raw_text=v_raw,
                parsed=v_parsed,
                parse_error=v_parse_error,
                verdict=verdict,
            )
        except Exception as exc:
            mep.errors.append(f"verifier: {exc}")
            mep.verifier = MEPVerifier(
                prompt="", raw_text="", verdict="skipped"
            )
    else:
        mep.verifier = MEPVerifier(prompt="", raw_text="", verdict="skipped")

    # ── Timestamps ────────────────────────────────────────────────────────
    run_end = _now_iso()
    mep.timestamps = MEPTimestamps(
        start=run_start,
        end=run_end,
        planner_ms=planner_ms,
        schema_retriever_ms=schema_ms,
        sql_generator_ms=sql_ms,
        verifier_ms=verifier_ms,
    )

    return mep


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RBC Metrics Eval Runner")

    # ── Input / output ────────────────────────────────────────────────────
    parser.add_argument(
        "--samples",
        default="eval_samples.json",
        help="Path to eval_samples.json",
    )
    parser.add_argument(
        "--mep_dir",
        required=True,
        help="Directory to write MEP JSON files",
    )

    # ── Database ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to UCI credit card CSV — loads into SQLite if provided",
    )
    parser.add_argument(
        "--db_uri",
        default=None,
        help="SQLAlchemy DB URI — used directly if --csv is not provided "
             "(e.g. sqlite:///rbc_metrics.db)",
    )
    parser.add_argument(
        "--db_path",
        default="rbc_metrics.db",
        help="SQLite file path when --csv is used (default: rbc_metrics.db)",
    )

    # ── Backend ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--backend",
        default="anthropic",
        choices=["openai", "gemini", "anthropic"],
        help="LLM backend for all agents",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Model name for the chosen backend",
    )
    parser.add_argument(
        "--config_name",
        default=None,
        help="Human-readable config label written into MEP "
             "(defaults to '<backend>_<model>')",
    )

    # ── Pipeline toggles ──────────────────────────────────────────────────
    parser.add_argument(
        "--no_schema",
        action="store_true",
        help="Skip SchemaRetrieverTool step",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip VerifierAgent step",
    )

    # ── Run controls ──────────────────────────────────────────────────────
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit to first N samples (default: run all)",
    )

    args = parser.parse_args()

    # ── Validate DB args ──────────────────────────────────────────────────
    if not args.csv and not args.db_uri:
        parser.error("One of --csv or --db_uri is required.")

    # ── DB setup — must run before agents are initialised ─────────────────
    if args.csv:
        db_uri = setup_db(args.csv, db_path=args.db_path)
    else:
        db_uri = args.db_uri

    # ── Run metadata ──────────────────────────────────────────────────────
    run_id      = str(uuid.uuid4())[:8]
    config_name = args.config_name or f"{args.backend}_{args.model}"
    config      = _make_config(
        backend=args.backend,
        model=args.model,
        config_name=config_name,
    )
    config.schema_retriever_enabled = not args.no_schema
    config.verifier_enabled         = not args.no_verify

    print(f"{'='*52}")
    print(f"  RBC Metrics Eval Runner")
    print(f"{'='*52}")
    print(f"  Run ID      : {run_id}")
    print(f"  Backend     : {args.backend} / {args.model}")
    print(f"  Config name : {config_name}")
    print(f"  DB URI      : {db_uri}")
    print(f"  MEP dir     : {args.mep_dir}")
    print(f"  Schema step : {'enabled' if config.schema_retriever_enabled else 'SKIPPED (--no_schema)'}")
    print(f"  Verifier    : {'enabled' if config.verifier_enabled else 'SKIPPED (--no_verify)'}")
    print(f"{'='*52}\n")

    # ── Agents — instantiate once, reuse across all samples ───────────────
    print("Initialising agents...")
    planner          = PlannerAgent(backend=args.backend, model=args.model)
    schema_retriever = SchemaRetrieverTool(db_uri=db_uri)
    sql_generator    = SQLGeneratorAgent(
        db_uri=db_uri,
        backend=args.backend,
        model=args.model,
    )
    verifier         = VerifierAgent(backend=args.backend, model=args.model)
    lf_client        = get_client()

    if lf_client:
        print("Langfuse        : enabled")
    else:
        print("Langfuse        : not configured "
              "(set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY to enable)")

    # ── Stub eval result files ────────────────────────────────────────────
    print(f"\nPreparing eval stub files in {args.mep_dir}...")
    eval_stub_paths = _write_stubs(args.mep_dir)
    for key, path in eval_stub_paths.items():
        print(f"  {key:15s} → {path}")

    # ── Load eval samples ─────────────────────────────────────────────────
    samples = load_eval_samples(args.samples)
    if args.n:
        samples = samples[: args.n]
    print(f"\nRunning {len(samples)} samples...\n")

    # ── Eval loop ─────────────────────────────────────────────────────────
    Path(args.mep_dir).mkdir(parents=True, exist_ok=True)
    success = error = citation_failures = guardrail_hits = 0

    for i, sample_dict in enumerate(samples, 1):
        sid = sample_dict.get("sample_id", f"sample_{i}")
        print(
            f"[{i:02d}/{len(samples)}] {sid} "
            f"({sample_dict.get('question_type', '?')}) — "
            f"{sample_dict['question'][:60]}..."
        )
        try:
            mep = run_sample(
                sample_dict,
                planner=planner,
                schema_retriever=schema_retriever,
                sql_generator=sql_generator,
                verifier=verifier,
                config=config,
                lf_client=lf_client,
                run_id=run_id,
            )
            mep_path = write_mep(mep, args.mep_dir)

            # ── Per-sample status line ────────────────────────────────
            verdict  = mep.verifier.verdict if mep.verifier else "skipped"
            cited    = bool(
                mep.sql_generator and mep.sql_generator.source_tables
            )
            guardrail = bool(
                mep.sql_generator and mep.sql_generator.guardrail_triggered
            )

            status_parts = [
                f"verdict={verdict}",
                f"citation={'✓' if cited else '✗'}",
            ]
            if guardrail:
                status_parts.append("GUARDRAIL")
            if mep.errors:
                status_parts.append(f"errors={mep.errors}")

            print(f"  {'✓' if not mep.errors else '⚠'}  {' | '.join(status_parts)}")
            print(f"     → {mep_path}")

            # ── Run-level counters ────────────────────────────────────
            success += 1
            if not cited:
                citation_failures += 1
            if guardrail:
                guardrail_hits += 1

        except Exception as exc:
            print(f"  ✗  FAILED: {exc}")
            error += 1

    # ── Run summary ───────────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"  Run complete")
    print(f"{'='*52}")
    print(f"  Total samples   : {len(samples)}")
    print(f"  Succeeded       : {success}")
    print(f"  Failed          : {error}")
    print(f"  Citation gaps   : {citation_failures}  "
          f"(source_tables empty — check sql_generator prompt)")
    print(f"  Guardrail hits  : {guardrail_hits}  "
          f"(queries blocked by guardrail rules)")
    print(f"\n  MEPs written to : {args.mep_dir}")

    # ── Next-step commands ────────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print("  Next: run eval passes in any order\n")
    print(
        f"  python -m rbc_metrics_eval.eval.eval_outputs \\\n"
        f"      --mep_dir {args.mep_dir} \\\n"
        f"      --out {eval_stub_paths['eval_outputs']}\n"
    )
    print(
        f"  python -m rbc_metrics_eval.eval.eval_traces \\\n"
        f"      --mep_dir {args.mep_dir} \\\n"
        f"      --out {eval_stub_paths['eval_traces']}\n"
    )
    print(
        f"  python -m rbc_metrics_eval.eval.eval_topk \\\n"
        f"      --mep_dir {args.mep_dir} \\\n"
        f"      --out {eval_stub_paths['eval_topk']} \\\n"
        f"      --backend {args.backend} \\\n"
        f"      --model {args.model}\n"
    )
    print(
        f"  python -m rbc_metrics_eval.eval.summarize \\\n"
        f"      --metrics {eval_stub_paths['eval_outputs']} \\\n"
        f"      --out summary.csv"
    )
    print(f"{'─'*52}\n")


if __name__ == "__main__":
    main()