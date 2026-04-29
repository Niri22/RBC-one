"""MEP file I/O utilities."""

import json
from pathlib import Path
from typing import Iterator

from .schema import MEP
from .schema import MEPSample
from .schema import MEPConfig
from .schema import MEPPlan
from .schema import ToolTrace
from .schema import MEPSchemaRetriever
from .schema import MEPSQLGenerator
from .schema import MEPVerifier



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

def init_mep(sample: MEPSample, config: MEPConfig, run_id: str) -> MEP:
    """Create a fresh MEP with metadata. Called at pipeline entry point."""

def append_schema(mep: MEP, schema_result: MEPSchemaRetriever) -> MEP:
    """Attach schema retriever output to MEP. Called after SchemaRetrieverTool."""

def append_sql(mep: MEP, sql_result: MEPSQLGenerator) -> MEP:
    """Attach SQL generator output. Validates source_tables is non-empty."""
    # Should raise or set error if source_tables is empty — citation requirement

def append_plan(mep: MEP, plan: MEPPlan) -> MEP:
    """Attach planner output to MEP."""

def append_verifier(mep: MEP, verifier: MEPVerifier) -> MEP:
    """Attach verifier output. Sets verdict field."""

def close_mep(mep: MEP, end_ts: str) -> MEP:
    """Set end timestamp and compute elapsed_ms per step. Called at pipeline exit."""

def validate_citation(mep: MEP) -> bool:
    """Return True if sql_generator.source_tables is non-empty. Used in eval_outputs."""

