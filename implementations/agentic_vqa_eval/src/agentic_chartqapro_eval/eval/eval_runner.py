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
    parser.add_argument("--samples",    required=True,  help="Path to eval_samples.json")
    parser.add_argument("--mep_dir",    required=True,  help="Directory to write MEP JSON files")
    parser.add_argument("--db_uri",     required=True,  help="SQLAlchemy DB URI for NL2SQLTool")
    parser.add_argument("--backend",    default="anthropic", choices=["openai", "gemini", "anthropic"])
    parser.add_argument("--model",      default="claude-sonnet-4-6")
    parser.add_argument("--config_name", default="anthropic_claude")
    parser.add_argument("--no_schema",  action="store_true", help="Skip SchemaRetrieverTool")
    parser.add_argument("--no_verify",  action="store_true", help="Skip VerifierAgent")
    parser.add_argument("--n",          type=int, default=None, help="Limit to first N samples")
    args = parser.parse_args()

    # ── Shared resources — init once ──────────────────────────────────────
    run_id = str(uuid.uuid4())[:8]
    config = _make_config(args.backend, args.model, args.config_name)
    config.schema_retriever_enabled = not args.no_schema
    config.verifier_enabled = not args.no_verify

    print(f"Run ID     : {run_id}")
    print(f"Backend    : {args.backend} / {args.model}")
    print(f"Schema step: {'enabled' if config.schema_retriever_enabled else 'SKIPPED'}")
    print(f"Verifier   : {'enabled' if config.verifier_enabled else 'SKIPPED'}")

    # Instantiate once — see SQLGeneratorAgent note on connection reuse
    planner        = PlannerAgent(backend=args.backend, model=args.model)
    schema_retriever = SchemaRetrieverTool(db_uri=args.db_uri)
    sql_generator  = SQLGeneratorAgent(db_uri=args.db_uri, backend=args.backend, model=args.model)
    verifier       = VerifierAgent(backend=args.backend, model=args.model)
    lf_client      = get_client()

    # ── Stub eval result files ────────────────────────────────────────────
    eval_stub_paths = _write_stubs(args.mep_dir)
    print(f"\nEval stub files written to {args.mep_dir}:")
    for k, p in eval_stub_paths.items():
        print(f"  {k:15s} → {p}")

    # ── Eval loop ─────────────────────────────────────────────────────────
    samples = load_eval_samples(args.samples)
    if args.n:
        samples = samples[: args.n]

    Path(args.mep_dir).mkdir(parents=True, exist_ok=True)
    success = error = 0

    print(f"\nRunning {len(samples)} samples...\n")
    for i, sample_dict in enumerate(samples, 1):
        sid = sample_dict.get("sample_id", f"sample_{i}")
        print(f"[{i:02d}/{len(samples)}] {sid} — {sample_dict['question'][:60]}...")
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

            # Surface any errors immediately
            if mep.errors:
                print(f"  ⚠ errors: {mep.errors}")
            else:
                verdict = mep.verifier.verdict if mep.verifier else "skipped"
                cited   = bool(mep.sql_generator and mep.sql_generator.source_tables)
                print(f"  ✓ verdict={verdict}  citation={'✓' if cited else '✗'}  → {mep_path}")

            success += 1
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")
            error += 1

    # ── Run summary ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Run complete — {success} ok, {error} failed")
    print(f"MEPs written to : {args.mep_dir}")
    print(f"\nNext steps:")
    print(f"  eval_outputs → python -m rbc_metrics_eval.eval.eval_outputs --mep_dir {args.mep_dir} --out {eval_stub_paths['eval_outputs']}")
    print(f"  eval_traces  → python -m rbc_metrics_eval.eval.eval_traces  --mep_dir {args.mep_dir} --out {eval_stub_paths['eval_traces']}")
    print(f"  eval_topk    → python -m rbc_metrics_eval.eval.eval_topk    --mep_dir {args.mep_dir} --out {eval_stub_paths['eval_topk']}")
    print(f"  summarize    → python -m rbc_metrics_eval.eval.summarize    --metrics {eval_stub_paths['eval_outputs']} --out summary.csv")


if __name__ == "__main__":
    main()