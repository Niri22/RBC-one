"""MEP file I/O utilities and builder functions.

Builder functions (init_mep → append_* → close_mep) are designed to be called
in sequence by the pipeline runner. They mutate and return the MEP so the
runner can chain calls without keeping extra state.
"""

import json
from pathlib import Path
from typing import Iterator

from ..utils.timing import iso_now
from .schema import (
    MEP,
    MEPConfig,
    MEPPlan,
    MEPSample,
    MEPSchemaRetriever,
    MEPSQLGenerator,
    MEPTimestamps,
    MEPVerifier,
    ToolTrace,
)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def write_mep(mep: MEP, out_dir: str) -> str:
    """Serialize MEP to JSON and write to <out_dir>/<sample_id>.json. Returns path."""
    path = Path(out_dir) / f"{mep.sample.sample_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        json.dump(mep.to_dict(), f, indent=2, default=str)
    return str(path)


def read_mep(path: str) -> dict:
    """Read a MEP JSON file from disk and return its content as a dict."""
    with open(path) as f:
        return json.load(f)


def iter_meps(mep_dir: str) -> Iterator[dict]:
    """Yield all MEP dicts from a directory, sorted by filename."""
    for p in sorted(Path(mep_dir).glob("*.json")):
        try:
            yield read_mep(str(p))
        except Exception as e:
            print(f"Warning: could not read MEP {p}: {e}")


# ---------------------------------------------------------------------------
# Builder functions — call in order: init → append_* → close
# ---------------------------------------------------------------------------


def init_mep(sample: MEPSample, config: MEPConfig, run_id: str) -> MEP:
    """Create a fresh MEP with metadata and start timestamp.

    Called at pipeline entry — before any agent runs.
    """
    return MEP(
        run_id=run_id,
        config=config,
        sample=sample,
        timestamps=MEPTimestamps(start=iso_now(), end=""),
    )


def append_plan(mep: MEP, plan: MEPPlan) -> MEP:
    """Attach planner output to MEP. Called after PlannerAgent.run()."""
    mep.plan = plan
    return mep


def append_schema(mep: MEP, schema_result: MEPSchemaRetriever) -> MEP:
    """Attach schema retriever output to MEP. Called after SchemaRetrieverTool.run()."""
    mep.schema_retriever = schema_result
    return mep


def append_sql(mep: MEP, sql_result: MEPSQLGenerator) -> MEP:
    """Attach SQL generator output. Sets a pipeline error if citation is missing.

    The citation requirement (non-empty source_tables) is enforced here so
    eval passes downstream can trust that any MEP without an error has cited
    its sources.
    """
    mep.sql_generator = sql_result
    if not sql_result.source_tables:
        mep.errors.append("citation_missing: sql_generator.source_tables is empty")
    return mep


def append_verifier(mep: MEP, verifier: MEPVerifier) -> MEP:
    """Attach verifier output and surface the verdict on the top-level object."""
    mep.verifier = verifier
    return mep


def close_mep(mep: MEP, end_ts: str) -> MEP:
    """Set end timestamp and compute per-step elapsed_ms from tool traces.

    Called at pipeline exit — after all agents have run or errored out.
    ``end_ts`` should be an ISO-format UTC string from ``iso_now()``.
    """
    if mep.timestamps is None:
        mep.timestamps = MEPTimestamps(start=end_ts, end=end_ts)
    mep.timestamps.end = end_ts

    # Pull elapsed_ms from tool traces where available
    def _trace_ms(trace_list) -> float:
        if not trace_list:
            return 0.0
        return sum(
            t.get("elapsed_ms", 0.0) if isinstance(t, dict) else getattr(t, "elapsed_ms", 0.0)
            for t in trace_list
        )

    if mep.plan:
        # PlannerAgent doesn't use a ToolTrace but timestamps.planner_ms is set by runner
        pass
    if mep.schema_retriever:
        mep.timestamps.schema_retriever_ms = _trace_ms(mep.schema_retriever.tool_trace)
    if mep.sql_generator:
        mep.timestamps.sql_generator_ms = _trace_ms(mep.sql_generator.tool_trace)

    return mep


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_citation(mep: MEP) -> bool:
    """Return True if sql_generator.source_tables is non-empty."""
    if mep.sql_generator is None:
        return False
    return len(mep.sql_generator.source_tables) > 0
