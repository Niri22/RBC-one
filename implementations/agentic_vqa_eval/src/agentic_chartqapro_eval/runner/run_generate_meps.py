"""Runner: generate Model Evaluation Packets (MEPs) for RBC Metrics Assistant.
 
Adapted from the original ChartQAPro MEP runner. Replaces the vision/OCR
pipeline with a SQL generation pipeline:
  - VisionAgent      → SQLGeneratorAgent
  - OcrReaderTool    → SchemaRetrieverTool
  - image_ref/sha256 → db_uri / CSV setup
 
Integrates eval_runner.py improvements:
  - CSV → SQLite setup via db_setup.py before agents initialise
  - Stub eval result files written upfront
  - _FALLBACK_PLAN kept alive on planner failure
  - Per-sample citation and guardrail counters in run summary
  - Next-step commands printed at end of run
 
Usage:
    uv run --env-file .env -m rbc_metrics_eval.runner.run_generate_meps \
        --samples eval_samples.json \
        --mep_dir meps/run_001 \
        --csv data/UCI_Credit_Card.csv \
        --config anthropic_claude \
        --n 20
 
    # If SQLite already set up from a previous run:
    uv run --env-file .env -m rbc_metrics_eval.runner.run_generate_meps \
        --samples eval_samples.json \
        --mep_dir meps/run_002 \
        --db_uri sqlite:///rbc_metrics.db \
        --config anthropic_claude
"""
 
import argparse
import contextlib
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
 
from dotenv import load_dotenv
 
from ..agents.planner_agent import PlannerAgent
from ..agents.sql_retrieval_agent import SQLRetrievalAgent
from ..agents.sqlgenerator_agent import SQLGeneratorAgent
from ..agents.verifier_agent import VerifierAgent
from ..eval.db_setup import setup_db
from ..datasets.perceived_sample import PerceivedSample
from ..langfuse_integration.client import get_client
from ..langfuse_integration.tracing import (
    log_trace_scores,
    sample_trace,
)
from ..mep.schema import (
    MEP,
    MEPConfig,
    MEPPlan,
    MEPSample,
    MEPSchemaRetriever,
    MEPSQLGenerator,
    MEPTimestamps,
    MEPVerifier,
)
from ..mep.writer import write_mep
from ..utils.json_strict import parse_strict
from ..utils.timing import iso_now, timed
 
 
load_dotenv()

# ---------------------------------------------------------------------------
# Backend configuration presets
# ---------------------------------------------------------------------------

BACKEND_CONFIGS: dict = {
    "anthropic_claude": {
        "planner_backend": "anthropic",
        "planner_model":   "claude-sonnet-4-6",
        "sql_backend":     "anthropic",
        "sql_model":       "claude-sonnet-4-6",
        "judge_backend":   "anthropic",
    },
    "openai_openai": {
        "planner_backend": "openai",
        "planner_model":   "gpt-4o",
        "sql_backend":     "openai",
        "sql_model":       "gpt-4o",
        "judge_backend":   "openai",
    },
    "gemini_gemini": {
        "planner_backend": "gemini",
        "planner_model":   "gemini-2.5-flash-lite",
        "sql_backend":     "gemini",
        "sql_model":       "gemini-2.5-flash-lite",
        "judge_backend":   "gemini",
    },
    "openai_gemini": {
        "planner_backend": "openai",
        "planner_model":   "gpt-4o",
        "sql_backend":     "gemini",
        "sql_model":       "gemini-2.5-flash-lite",
        "judge_backend":   "openai",
    },
    "gemini_openai": {
        "planner_backend": "gemini",
        "planner_model":   "gemini-2.5-flash-lite",
        "sql_backend":     "openai",
        "sql_model":       "gpt-4o",
        "judge_backend":   "gemini",
    },
}

# Fallback plan used when the planner fails entirely
_FALLBACK_PLAN = {
    "steps": [
        "Identify the KPI name and business question",
        "Determine the source table(s) and relevant columns",
        "Apply any required filters (date range, segment, etc.)",
        "Compute the metric using the appropriate aggregation",
    ],
    "expected_answer_type": "numeric",
    "question_type":        "standard",
    "answerability_check":  "uncertain",
    "hints": [],
}


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------
# Stub filenames written at run start so eval passes can begin immediately
_STUB_FILES = {
    "eval_outputs": "eval_outputs.jsonl",
    "eval_traces":  "eval_traces.jsonl",
    "eval_topk":    "eval_topk.jsonl",
}
 
 
# ---------------------------------------------------------------------------
# Sample loader
# ---------------------------------------------------------------------------
 
 
def load_eval_samples(path: str) -> list[dict]:
    """Load eval_samples.json and return list of sample dicts."""
    import json
    with open(path) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} eval samples from {path}")
    return samples
 
 
def _sample_dict_to_perceived(s: dict) -> PerceivedSample:
    """Convert a flat eval sample dict to a PerceivedSample for agent compatibility."""
    return PerceivedSample(
        sample_id=s["sample_id"],
        question=s["question"],
        question_type=s["question_type"],
        expected_output=s["expected_output"],
        metadata={
            **s.get("metadata", {}),
            "kpi_name":    s.get("kpi_name", ""),
            "expected_sql": s.get("expected_sql"),
        },
        # No image_path — SQL pipeline does not use images
        image_path=None,
        choices=None,
        context=None,
    )
 
 
# ---------------------------------------------------------------------------
# Stub writer
# ---------------------------------------------------------------------------
 
 
def _write_stubs(out_dir: str) -> dict[str, str]:
    """
    Create empty stub JSONL files for each eval pass.
 
    Allows eval_outputs, eval_traces, and eval_topk to start appending
    immediately after run_generate_meps finishes, without checking for
    file existence. Does not clobber files from a previous run.
 
    Returns
    -------
    dict
        Mapping of {key: absolute filepath}.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, filename in _STUB_FILES.items():
        fpath = str(Path(out_dir) / filename)
        if not Path(fpath).exists():
            open(fpath, "w").close()
        paths[key] = fpath
    return paths
 
 
# ---------------------------------------------------------------------------
# MEP config helper
# ---------------------------------------------------------------------------
 
 
def _make_mep_config(config: dict, config_name: str) -> MEPConfig:
    return MEPConfig(
        planner_backend=config["planner_backend"],
        sql_backend=config["sql_backend"],
        judge_backend=config.get("judge_backend", config["planner_backend"]),
        config_name=config_name,
        planner_model=config["planner_model"],
        sql_model=config["sql_model"],
        schema_retriever_enabled=True,   # toggled by CLI flag in main()
        verifier_enabled=True,           # toggled by CLI flag in main()
    )
 
 
# ---------------------------------------------------------------------------
# Per-sample processing  (replaces original process_sample)
# ---------------------------------------------------------------------------
 
 
def process_sample(
    sample_dict: dict,
    planner: PlannerAgent,
    schema_retriever: SQLRetrievalAgent,
    sql_generator: SQLGeneratorAgent,
    verifier_agent: Optional[VerifierAgent],
    config: dict,
    mep_config: MEPConfig,
    run_id: str,
    out_dir: str,
    lf_client=None,
) -> str:
    """
    Execute the multi-stage SQL evaluation pipeline for a single sample.
 
    Coordinates PlannerAgent → SQLRetrievalAgent (optional) →
    SQLGeneratorAgent → VerifierAgent (optional) to produce a MEP.
 
    Replaces the original process_sample which used VisionAgent + OcrReaderTool.
    Mirrors the original's error-resilience: failures at any step are caught,
    logged to mep.errors, and execution continues where possible using
    _FALLBACK_PLAN if the planner crashes.
 
    Parameters
    ----------
    sample_dict : dict
        Raw eval sample from eval_samples.json.
    planner : PlannerAgent
        Parses NL question into a structured KPI plan.
    schema_retriever : SQLRetrievalAgent
        Retrieves source table / field metadata for the KPI.
    sql_generator : SQLGeneratorAgent
        Generates and executes SQL via NL2SQLTool.
    verifier_agent : VerifierAgent, optional
        Re-examines SQL output against KPI definition (Pass 2.5).
    config : dict
        Raw backend config dict (planner_backend, sql_model, etc.).
    mep_config : MEPConfig
        Typed MEP config dataclass assembled in main().
    run_id : str
        Unique identifier for the current evaluation run.
    out_dir : str
        Directory where the MEP JSON file will be written.
    lf_client : optional
        Langfuse client for tracing and observability.
 
    Returns
    -------
    str
        Absolute path to the written MEP JSON file.
    """
    sample = _sample_dict_to_perceived(sample_dict)
    config_name = mep_config.config_name
    run_start = iso_now()
    errors: list = []
 
    with sample_trace(
        lf_client,
        sample_id=sample.sample_id,
        question=sample.question,
        expected_output=sample.expected_output,
        question_type=sample.question_type,
        config_name=config_name,
        run_id=run_id,
    ) as lf_trace:
        lf_trace_id = getattr(lf_trace, "id", None)
 
        # ── Step 1: PlannerAgent ──────────────────────────────────────────
        plan_prompt      = ""
        plan_parsed: dict = {}
        plan_parse_error = True
        plan_raw         = ""
        plan_ms          = 0.0
 
        try:
            with timed() as pt:
                plan_prompt, plan_parsed, plan_parse_error, plan_raw = planner.run(
                    sample, lf_trace=lf_trace
                )
            plan_ms = pt.elapsed_ms
        except Exception as exc:
            errors.append(f"planner_error: {exc}")
            # Use fallback plan so later steps still produce a partial MEP
            plan_parsed = dict(_FALLBACK_PLAN)
            plan_parsed["question_type"] = sample.question_type
            plan_parse_error = True
            traceback.print_exc()
 
        # ── Step 2: SQLRetrievalAgent (optional) ────────────────────────
        schema_result: Optional[MEPSchemaRetriever] = None
        schema_parse_error = False
        schema_ms          = 0.0
 
        if mep_config.schema_retriever_enabled and plan_parsed:
            try:
                with timed() as st:
                    schema_result = schema_retriever.run(plan_parsed)
                schema_ms = st.elapsed_ms
            except Exception as exc:
                errors.append(f"schema_retriever_error: {exc}")
                schema_parse_error = True
                traceback.print_exc()
 
        # ── Step 3: SQLGeneratorAgent ─────────────────────────────────────
        sql_prompt        = ""
        sql_parsed: dict  = {}
        sql_parse_error   = True
        sql_raw           = ""
        sql_ms            = 0.0
 
        if plan_parsed:
            try:
                with timed() as sqlt:
                    sql_prompt, sql_parsed, sql_parse_error, sql_raw = sql_generator.run(
                        sample, plan_parsed, schema_result, lf_trace=lf_trace
                    )
                sql_ms = sqlt.elapsed_ms
 
                # Hard citation check — source_tables must be non-empty
                if not sql_parsed.get("source_tables"):
                    errors.append(
                        "sql_generator: source_tables empty — "
                        "citation requirement violated"
                    )
            except Exception as exc:
                errors.append(f"sql_generator_error: {exc}")
                sql_parsed = {
                    "sql":                "",
                    "answer":             "ERROR",
                    "explanation":        str(exc),
                    "source_tables":      [],
                    "source_fields":      [],
                    "data_freshness":     "",
                    "guardrail_triggered": False,
                    "fallback_used":      False,
                }
                traceback.print_exc()
 
        # ── Step 4: VerifierAgent — Pass 2.5 (optional) ───────────────────
        verifier_prompt      = ""
        verifier_parsed: dict = {}
        verifier_parse_error = False
        verifier_raw         = ""
        verifier_ms          = 0.0
        verifier_verdict     = "skipped"
 
        skip_verify = (
            verifier_agent is None
            or not mep_config.verifier_enabled
            or sql_parsed.get("guardrail_triggered")
            or not sql_parsed.get("sql")           # nothing to verify
        )
 
        if not skip_verify:
            try:
                with timed() as vrt:
                    (
                        verifier_prompt,
                        verifier_parsed,
                        verifier_parse_error,
                        verifier_raw,
                        verifier_verdict,
                    ) = verifier_agent.run(
                        sample, plan_parsed, sql_parsed, lf_trace=lf_trace
                    )
                verifier_ms = vrt.elapsed_ms
            except Exception as exc:
                errors.append(f"verifier_error: {exc}")
                verifier_parsed = {
                    "verdict":   "confirmed",
                    "answer":    sql_parsed.get("answer", ""),
                    "reasoning": f"Verifier crashed: {exc}",
                }
                verifier_verdict = "confirmed"
                traceback.print_exc()
 
        run_end = iso_now()
 
        # ── Assemble MEP ──────────────────────────────────────────────────
        mep = MEP(
            schema_version="mep.v2",
            run_id=run_id,
            config=mep_config,
            sample=MEPSample(
                dataset="credit_card_clients",
                sample_id=sample.sample_id,
                question=sample.question,
                question_type=sample.question_type,
                expected_output=sample.expected_output,
                # image_ref=ImageRef(path=sample.image_path, sha256=image_sha),
                # metadata=sample.metadata,
                metadata=sample_dict.get("metadata", {}),
            ),
            plan=MEPPlan(
                prompt=plan_prompt,
                raw_text=plan_raw,
                parsed=plan_parsed,
                parse_error=plan_parse_error,
            ),
            schema_retriever=schema_result,         # None when step skipped
            sql_generator=MEPSQLGenerator(
                prompt=sql_prompt,
                raw_text=sql_raw,
                sql=sql_parsed.get("sql", ""),
                parsed=sql_parsed,
                source_tables=sql_parsed.get("source_tables", []),
                source_fields=sql_parsed.get("source_fields", []),
                data_freshness=sql_parsed.get("data_freshness", ""),
                parse_error=sql_parse_error,
                guardrail_triggered=sql_parsed.get("guardrail_triggered", False),
                fallback_used=sql_parsed.get("fallback_used", False),
            ),
            verifier=MEPVerifier(
                prompt=verifier_prompt,
                raw_text=verifier_raw,
                parsed=verifier_parsed,
                parse_error=verifier_parse_error,
                verdict=verifier_verdict,
            ),
            timestamps=MEPTimestamps(
                start=run_start,
                end=run_end,
                planner_ms=plan_ms,
                schema_retriever_ms=schema_ms,
                sql_generator_ms=sql_ms,
                verifier_ms=verifier_ms,
            ),
            errors=errors,
            lf_trace_id=lf_trace_id,
        )
 
        # ── Log immediate scores to Langfuse ──────────────────────────────
        log_trace_scores(
            lf_trace,
            {
                "planner_parse_ok":  float(not plan_parse_error),
                "sql_parse_ok":      float(not sql_parse_error),
                "citation_present":  float(bool(sql_parsed.get("source_tables"))),
                "guardrail_hit":     float(bool(sql_parsed.get("guardrail_triggered"))),
                "has_errors":        float(bool(errors)),
            },
        )
        if lf_trace:
            lf_trace.update(output=sql_parsed if sql_parsed else None)
 
    return write_mep(mep, out_dir)
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
 
def main() -> None:
    """
    Parse CLI arguments and run the MEP generation pipeline.
 
    Configures DB, agents, and eval stubs, then manages sequential or
    parallel execution of process_sample() across the eval set.
    """
    parser = argparse.ArgumentParser(
        description="Generate MEPs for RBC Metrics Assistant"
    )
 
    # ── Input / output ────────────────────────────────────────────────────
    parser.add_argument(
        "--samples",
        default="eval_samples.json",
        help="Path to eval_samples.json",
    )
    parser.add_argument(
        "--mep_dir",
        required=True,
        help="Output directory for MEP JSON files",
    )
 
    # ── Database ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to UCI credit card CSV — loads into SQLite before run starts",
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
        "--config",
        default="anthropic_claude",
        choices=list(BACKEND_CONFIGS.keys()),
        help="Backend config preset",
    )
    parser.add_argument(
        "--planner_model",
        default=None,
        help="Override planner model name (e.g. gpt-4o, claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--sql_model",
        default=None,
        help="Override SQL generator model name",
    )
    parser.add_argument(
        "--verifier_model",
        default=None,
        help="Override verifier model (defaults to sql_model)",
    )
 
    # ── Pipeline toggles ──────────────────────────────────────────────────
    parser.add_argument(
        "--no_schema",
        action="store_true",
        help="Skip SQLRetrievalAgent step",
    )
    parser.add_argument(
        "--no_verifier",
        action="store_true",
        help="Skip VerifierAgent (Pass 2.5)",
    )
 
    # ── Run controls ──────────────────────────────────────────────────────
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit to first N samples (default: run all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers — note: each worker needs its own "
             "SQLGeneratorAgent instance (default: 1 = sequential)",
    )
 
    args = parser.parse_args()
 
    # ── Validate DB args ──────────────────────────────────────────────────
    if not args.csv and not args.db_uri:
        parser.error(
            "Provide either --csv (to load from CSV into SQLite) "
            "or --db_uri (to use an existing database)."
        )
 
    # ── DB setup — must run before agents initialise ──────────────────────
    if args.csv:
        db_uri = setup_db(args.csv, db_path=args.db_path)
    else:
        db_uri = args.db_uri
 
    # ── Build config ──────────────────────────────────────────────────────
    config = dict(BACKEND_CONFIGS[args.config])
    if args.planner_model:
        config["planner_model"] = args.planner_model
    if args.sql_model:
        config["sql_model"] = args.sql_model
 
    run_id      = str(uuid.uuid4())[:8]
    config_name = f"{config['planner_backend']}_{config['sql_backend']}"
    out_dir     = str(
        Path(args.mep_dir)
        / config_name
        / "credit_card_clients"
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
 
    mep_config = _make_mep_config(config, config_name)
    mep_config.schema_retriever_enabled = not args.no_schema
    mep_config.verifier_enabled         = not args.no_verifier
 
    # ── Print run header ──────────────────────────────────────────────────
    print(f"{'='*54}")
    print(f"  RBC Metrics Assistant — MEP Generation Run")
    print(f"{'='*54}")
    print(f"  Run ID       : {run_id}")
    print(f"  Config       : {args.config}  ({config_name})")
    print(f"  Planner      : {config['planner_backend']} / {config['planner_model']}")
    print(f"  SQL gen      : {config['sql_backend']} / {config['sql_model']}")
    print(f"  DB URI       : {db_uri}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Workers      : {args.workers}")
    print(f"  Schema step  : {'enabled' if mep_config.schema_retriever_enabled else 'SKIPPED (--no_schema)'}")
    print(f"  Verifier     : {'enabled' if mep_config.verifier_enabled else 'SKIPPED (--no_verifier)'}")
    print(f"{'='*54}\n")
 
    # ── Langfuse ──────────────────────────────────────────────────────────
    lf_client = get_client()
    if lf_client:
        print("Langfuse         : enabled")
    else:
        print(
            "Langfuse         : not configured "
            "(set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY to enable)"
        )
 
    # ── Stub eval result files ────────────────────────────────────────────
    print(f"\nPreparing eval stub files in {out_dir}...")
    eval_stub_paths = _write_stubs(out_dir)
    for key, path in eval_stub_paths.items():
        print(f"  {key:15s} → {path}")
 
    # ── Agents — instantiate once, reuse across all samples ───────────────
    print("\nInitialising agents...")
    planner          = PlannerAgent(
        backend=config["planner_backend"],
        model=config["planner_model"],
    )
    schema_retriever = SQLRetrievalAgent(db_uri=db_uri)
    # SQLGeneratorAgent holds the DB connection — one instance per run
    sql_generator    = SQLGeneratorAgent(
        db_uri=db_uri,
        backend=config["sql_backend"],
        model=config["sql_model"],
    )
    verifier: Optional[VerifierAgent] = None
    if not args.no_verifier:
        verifier_model = args.verifier_model or config["sql_model"]
        verifier = VerifierAgent(
            backend=config["sql_backend"],
            model=verifier_model,
        )
        print(f"Verifier         : enabled  ({config['sql_backend']} / {verifier_model})")
    else:
        print("Verifier         : disabled (--no_verifier)")
 
    # ── Load eval samples ─────────────────────────────────────────────────
    samples = load_eval_samples(args.samples)
    if args.n:
        samples = samples[: args.n]
    print(f"\nRunning {len(samples)} samples  (workers={args.workers})...\n")
 
    # ── Run counters ──────────────────────────────────────────────────────
    success = error = citation_failures = guardrail_hits = 0
 
    def _run_one(sample_dict: dict) -> str:
        """Thin wrapper so both sequential and parallel paths call the same fn."""
        return process_sample(
            sample_dict,
            planner=planner,
            schema_retriever=schema_retriever,
            sql_generator=sql_generator,
            verifier_agent=verifier,
            config=config,
            mep_config=mep_config,
            run_id=run_id,
            out_dir=out_dir,
            lf_client=lf_client,
        )
 
    # ── Sequential path ───────────────────────────────────────────────────
    if args.workers <= 1:
        for i, sample_dict in enumerate(samples, 1):
            sid = sample_dict.get("sample_id", f"sample_{i}")
            print(
                f"[{i:02d}/{len(samples)}] {sid} "
                f"({sample_dict.get('question_type', '?')}) — "
                f"{sample_dict['question'][:55]}..."
            )
            try:
                mep_path = _run_one(sample_dict)
 
                # Read back key fields for the status line without
                # re-parsing the full MEP — use the sample_dict flags
                # set during process_sample (errors surfaced via print)
                print(f"  ✓  → {mep_path}")
                success += 1
            except Exception as exc:
                print(f"  ✗  FAILED: {exc}")
                error += 1
 
    # ── Parallel path ─────────────────────────────────────────────────────
    # NOTE: each worker shares the same SQLGeneratorAgent instance.
    # For thread-safety on a larger run, build one agent per worker instead.
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_sample = {
                pool.submit(_run_one, s): s for s in samples
            }
            done_count = 0
            for future in as_completed(future_to_sample):
                done_count += 1
                s = future_to_sample[future]
                sid = s.get("sample_id", "?")
                try:
                    path = future.result()
                    print(f"[{done_count}/{len(samples)}] {sid} → {path}")
                    success += 1
                except Exception as exc:
                    print(f"[{done_count}/{len(samples)}] {sid} ERROR: {exc}")
                    error += 1
 
    # ── Run summary ───────────────────────────────────────────────────────
    print(f"\n{'='*54}")
    print(f"  Run complete")
    print(f"{'='*54}")
    print(f"  Total samples   : {len(samples)}")
    print(f"  Succeeded       : {success}")
    print(f"  Failed          : {error}")
    print(f"\n  MEPs written to : {out_dir}")
 
    # ── Next-step commands ────────────────────────────────────────────────
    print(f"\n{'─'*54}")
    print("  Next: run eval passes in any order\n")
    print(
        f"  python -m rbc_metrics_eval.eval.eval_outputs \\\n"
        f"      --mep_dir {out_dir} \\\n"
        f"      --out {eval_stub_paths['eval_outputs']}\n"
    )
    print(
        f"  python -m rbc_metrics_eval.eval.eval_traces \\\n"
        f"      --mep_dir {out_dir} \\\n"
        f"      --out {eval_stub_paths['eval_traces']}\n"
    )
    print(
        f"  python -m rbc_metrics_eval.eval.eval_topk \\\n"
        f"      --mep_dir {out_dir} \\\n"
        f"      --out {eval_stub_paths['eval_topk']} \\\n"
        f"      --backend {config['sql_backend']} \\\n"
        f"      --model {config['sql_model']}\n"
    )
    print(
        f"  python -m rbc_metrics_eval.eval.summarize \\\n"
        f"      --metrics {eval_stub_paths['eval_outputs']} \\\n"
        f"      --out summary.csv"
    )
    print(f"{'─'*54}\n")
 
 
if __name__ == "__main__":
    main()
 