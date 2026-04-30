"""Programmatic API for running the SQL evaluation pipeline.

This module exposes ``run_sample`` and ``load_eval_samples`` as a clean
importable API for notebooks, tests, and scripts that need to drive the
pipeline without invoking the CLI.

For batch evaluation runs (the common case) use the CLI entry point instead:

    uv run --env-file .env -m agentic_chartqapro_eval.runner.run_generate_meps \\
        --samples eval_samples.json \\
        --mep_dir meps/run_001 \\
        --csv data/UCI_Credit_Card.csv

Responsibilities
----------------
run_generate_meps.py  CLI batch runner — argparse, parallel workers, DB setup,
                      stub file creation, run summary, next-step commands.
eval_runner.py        Programmatic API — importable ``run_sample`` and
                      ``load_eval_samples`` for interactive use.

Both modules share the same per-sample logic; ``run_generate_meps.process_sample``
is the canonical implementation.  The helpers here delegate to it so there is
no duplicated pipeline code.
"""

import json
import uuid
from typing import Any, Optional

from ..agents.planner_agent import PlannerAgent
from ..agents.sql_retrieval_agent import SQLRetrievalAgent
from ..agents.sqlgenerator_agent import SQLGeneratorAgent
from ..agents.verifier_agent import VerifierAgent
from ..mep.schema import MEP, MEPConfig
from ..runner.run_generate_meps import (
    _make_mep_config,
    process_sample,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def load_eval_samples(path: str) -> list[dict]:
    """Load eval_samples.json and return list of sample dicts."""
    with open(path) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} eval samples from {path}")
    return samples


def run_sample(
    sample_dict: dict,
    planner: PlannerAgent,
    schema_retriever: SQLRetrievalAgent,
    sql_generator: SQLGeneratorAgent,
    verifier: Optional[VerifierAgent],
    config: MEPConfig,
    lf_client: Any = None,
    run_id: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> MEP:
    """Run a single eval sample through the full SQL pipeline and return a MEP.

    Thin wrapper around ``run_generate_meps.process_sample`` that accepts a
    typed ``MEPConfig`` and returns the MEP object rather than writing it to
    disk (unless ``out_dir`` is provided).

    Parameters
    ----------
    sample_dict : dict
        Raw eval sample dict from eval_samples.json.
    planner : PlannerAgent
    schema_retriever : SQLRetrievalAgent
    sql_generator : SQLGeneratorAgent
    verifier : VerifierAgent or None
        Pass ``None`` to skip the verifier step.
    config : MEPConfig
        Pipeline configuration (schema_retriever_enabled, verifier_enabled, etc.)
    lf_client : optional
        Langfuse client for tracing.
    run_id : str, optional
        Unique run identifier — generated if not provided.
    out_dir : str, optional
        If given, the MEP is written to disk and the file path is printed.

    Returns
    -------
    MEP
        Fully populated MEP dataclass.
    """
    _run_id = run_id or str(uuid.uuid4())[:8]

    # Build the raw config dict that process_sample expects
    raw_config = {
        "planner_backend": config.planner_backend,
        "planner_model":   config.planner_model,
        "sql_backend":     config.sql_backend,
        "sql_model":       config.sql_model,
        "judge_backend":   config.judge_backend,
    }

    _out_dir = out_dir or "/tmp/meps"
    mep_path = process_sample(
        sample_dict,
        planner=planner,
        schema_retriever=schema_retriever,
        sql_generator=sql_generator,
        verifier_agent=verifier,
        config=raw_config,
        mep_config=config,
        run_id=_run_id,
        out_dir=_out_dir,
        lf_client=lf_client,
    )

    if out_dir:
        print(f"MEP written to {mep_path}")

    # Re-load from disk so the caller always gets a consistent MEP object
    import json as _json
    with open(mep_path) as f:
        mep_dict = _json.load(f)

    # Return the raw dict wrapped back into a MEP — or just the dict for now
    # (full round-trip deserialization would require MEP.from_dict)
    return mep_dict  # type: ignore[return-value]


def make_config(
    backend: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    config_name: Optional[str] = None,
    schema_retriever_enabled: bool = True,
    verifier_enabled: bool = True,
) -> MEPConfig:
    """Convenience factory for MEPConfig — avoids importing MEPConfig directly."""
    raw = {
        "planner_backend": backend,
        "planner_model":   model,
        "sql_backend":     backend,
        "sql_model":       model,
        "judge_backend":   "anthropic",
    }
    cfg = _make_mep_config(raw, config_name or f"{backend}_{model}")
    cfg.schema_retriever_enabled = schema_retriever_enabled
    cfg.verifier_enabled = verifier_enabled
    return cfg
