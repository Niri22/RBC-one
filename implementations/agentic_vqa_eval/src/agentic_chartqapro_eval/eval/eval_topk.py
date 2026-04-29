r"""Top-K answer evaluation pass for the SQL pipeline.

For each MEP, re-runs the SQL generator up to K times using a temperature
ladder (0.2 → 0.4 → 0.6) to produce diverse candidate answers backed by real
database execution. Computes hit@1, hit@2, hit@3 without modifying existing
MEPs or metrics.

Usage:
    uv run --env-file .env -m agentic_chartqapro_eval.eval.eval_topk \
        --mep_dir meps/openai_openai/sql/test \
        --out topk_metrics.jsonl \
        --db_uri sqlite:///rbc_metrics.db \
        --backend anthropic \
        --model claude-sonnet-4-6 \
        --k 3
"""

import argparse
import contextlib
import json
import os
from typing import List, Optional

from crewai import LLM, Agent, Crew, Task
from crewai_tools import NL2SQLTool
from dotenv import load_dotenv

from ..agents.sqlgenerator_agent import (
    SQL_REQUIRED_KEYS,
    _apply_guardrails,
    build_sql_generator_prompt,
)
from ..datasets.perceived_sample import PerceivedSample, QuestionType
from ..langfuse_integration.client import get_client
from ..mep.schema import MEPSchemaRetriever
from ..mep.writer import iter_meps
from ..utils.json_strict import parse_strict
from .eval_outputs import score_answer_accuracy


load_dotenv()

_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Temperature ladder — each attempt gets slightly more variation so candidates differ
_TEMPERATURES = [0.2, 0.4, 0.6]


def _build_llm(backend: str, model: str, api_key: Optional[str], temperature: float) -> LLM:
    env_key = _API_KEY_ENV.get(backend, "")
    key = api_key or os.environ.get(env_key, "")
    if backend == "openai":
        return LLM(model=model, api_key=key, temperature=temperature)
    if backend == "gemini":
        return LLM(model=f"gemini/{model}", api_key=key, temperature=temperature)
    if backend == "anthropic":
        return LLM(model=f"anthropic/{model}", api_key=key, temperature=temperature)
    raise ValueError(f"Unknown topk backend: {backend!r}")


def _mep_to_perceived_sample(mep: dict) -> PerceivedSample:
    s = mep.get("sample", {})
    try:
        qtype = QuestionType(s.get("question_type", "standard").lower())
    except ValueError:
        qtype = QuestionType.STANDARD
    return PerceivedSample(
        sample_id=s.get("sample_id", ""),
        image_path="",
        question=s.get("question", ""),
        expected_output=s.get("expected_output", ""),
        question_type=qtype,
        metadata=s.get("metadata", {}),
    )


def _mep_to_schema(mep: dict) -> Optional[MEPSchemaRetriever]:
    sr = mep.get("schema_retriever")
    if not sr:
        return None
    return MEPSchemaRetriever(
        kpi_name=sr.get("kpi_name", ""),
        source_tables=sr.get("source_tables", []),
        source_fields=sr.get("source_fields", []),
        join_keys=sr.get("join_keys", []),
        data_freshness=sr.get("data_freshness", ""),
    )


def _run_single_candidate(
    sample: PerceivedSample,
    plan: dict,
    schema: Optional[MEPSchemaRetriever],
    db_uri: str,
    backend: str,
    model: str,
    api_key: Optional[str],
    temperature: float,
) -> Optional[str]:
    """Run one SQL generation attempt and return the answer string, or None on failure."""
    prompt = build_sql_generator_prompt(sample, plan, schema)
    allowed_tables = schema.source_tables if schema else []

    llm = _build_llm(backend, model, api_key, temperature)
    nl2sql = NL2SQLTool(db_uri=db_uri)

    agent = Agent(
        role="SQL Metrics Generator",
        goal=(
            "Generate a precise SQL query that answers the KPI question. "
            "Output JSON only — no extra text."
        ),
        backstory=(
            "You are a senior data engineer. You write exact, reproducible SQL "
            "queries for business metrics. Never use SELECT * and always cite "
            "source tables."
        ),
        llm=llm,
        tools=[nl2sql],
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description=prompt,
        expected_output=(
            "A JSON object with keys: sql, answer, explanation, source_tables, "
            "source_fields, data_freshness, guardrail_triggered, fallback_used"
        ),
        agent=agent,
    )
    result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()

    raw = getattr(result, "raw", None) or str(result)
    parsed, _ = parse_strict(raw, required_keys=["answer", "sql"])
    if not parsed:
        return None

    triggered, _ = _apply_guardrails(parsed.get("sql", ""), allowed_tables)
    if triggered:
        return None

    answer = str(parsed.get("answer", "")).strip()
    return answer or None


def get_topk_candidates(
    mep: dict,
    db_uri: str,
    k: int = 3,
    backend: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
) -> List[str]:
    """Re-run SQLGeneratorAgent up to k times with a temperature ladder.

    Returns a deduplicated list of candidate answers (up to k items). Using
    temperature > 0 ensures each attempt can produce a different SQL path
    and therefore a different answer when the original was wrong.
    """
    sample = _mep_to_perceived_sample(mep)
    plan = mep.get("plan", {}).get("parsed", {})
    schema = _mep_to_schema(mep)

    candidates: List[str] = []
    seen: set = set()
    temperatures = (_TEMPERATURES + [_TEMPERATURES[-1]] * k)[:k]  # extend if k > 3

    for attempt, temp in enumerate(temperatures):
        try:
            answer = _run_single_candidate(
                sample, plan, schema, db_uri, backend, model, api_key, temp
            )
            if answer and answer.upper() not in seen:
                seen.add(answer.upper())
                candidates.append(answer)
                if len(candidates) >= k:
                    break
        except Exception as exc:
            sid = mep.get("sample", {}).get("sample_id", "?")
            print(f"  topk attempt {attempt + 1} error for {sid}: {exc}")

    return candidates


def _hit_at_k(expected: str, candidates: List[str], question_type: str, k: int) -> float:
    """1.0 if expected matches any of the first k candidates."""
    for c in candidates[:k]:
        if score_answer_accuracy(expected, c, question_type) > 0:
            return 1.0
    return 0.0


def evaluate_topk(
    mep: dict,
    db_uri: str,
    k: int = 3,
    backend: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
) -> dict:
    """Evaluate top-K SQL answer candidates for a single MEP."""
    sample = mep.get("sample", {})
    config = mep.get("config", {})
    sql = mep.get("sql_generator", {})

    expected = sample.get("expected_output", "")
    question_type = sample.get("question_type", "standard")
    original_answer = (
        (mep.get("verifier") or {}).get("parsed", {}).get("answer")
        or sql.get("parsed", {}).get("answer", "")
    )

    candidates = get_topk_candidates(
        mep, db_uri=db_uri, k=k, backend=backend, model=model, api_key=api_key
    )

    result: dict = {
        "sample_id": sample.get("sample_id", ""),
        "question_type": question_type,
        "config_name": config.get("config_name", ""),
        "expected": expected,
        "original_answer": original_answer,
        "topk_candidates": candidates,
        "original_accuracy": score_answer_accuracy(expected, original_answer, question_type),
        "citation_present": len(sql.get("source_tables", [])) > 0,
    }

    for ki in range(1, k + 1):
        result[f"hit_at_{ki}"] = _hit_at_k(expected, candidates, question_type, ki)

    return result


def main() -> None:
    """Run top-K evaluation on MEPs and write results to JSONL."""
    parser = argparse.ArgumentParser(description="Top-K answer candidate evaluation")
    parser.add_argument("--mep_dir", required=True)
    parser.add_argument("--out", default="topk_metrics.jsonl")
    parser.add_argument("--db_uri", default=None, help="SQLAlchemy DB URI")
    parser.add_argument("--csv", default=None, help="Path to CSV — auto-loaded into SQLite")
    parser.add_argument(
        "--backend", default="anthropic", choices=["openai", "gemini", "anthropic"]
    )
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=None, help="Limit to first N MEPs")
    args = parser.parse_args()

    if args.csv:
        from .db_setup import setup_db
        db_uri = setup_db(args.csv)
    elif args.db_uri:
        db_uri = args.db_uri
    else:
        raise SystemExit("error: --db_uri or --csv is required")

    api_key = os.environ.get(_API_KEY_ENV[args.backend], "")
    lf_client = get_client()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w") as f_out:
        count = 0
        for mep in iter_meps(args.mep_dir):
            if args.n is not None and count >= args.n:
                break
            try:
                result = evaluate_topk(
                    mep,
                    db_uri=db_uri,
                    k=args.k,
                    backend=args.backend,
                    model=args.model,
                    api_key=api_key,
                )
                f_out.write(json.dumps(result) + "\n")
                sid = result["sample_id"]
                exp = result["expected"]
                cands = result["topk_candidates"]
                h1 = result.get("hit_at_1", 0)
                hk = result.get(f"hit_at_{args.k}", 0)
                print(f"  {sid}  exp={exp!r}  candidates={cands}  hit@1={h1}  hit@{args.k}={hk}")

                lf_trace_id = mep.get("lf_trace_id")
                if lf_client and lf_trace_id:
                    for ki in range(1, args.k + 1):
                        key = f"hit_at_{ki}"
                        if key in result:
                            with contextlib.suppress(Exception):
                                lf_client.create_score(
                                    trace_id=lf_trace_id,
                                    name=key,
                                    value=float(result[key]),
                                )
                count += 1
            except Exception as exc:
                print(f"  Error: {exc}")

    with open(args.out) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if records:
        print(f"\n--- Top-K Summary (n={len(records)}) ---")
        orig_acc = sum(r["original_accuracy"] for r in records) / len(records)
        print(f"  Original accuracy (from MEP)  : {orig_acc:.3f}")
        for ki in range(1, args.k + 1):
            key = f"hit_at_{ki}"
            if key in records[0]:
                h = sum(r[key] for r in records) / len(records)
                print(f"  hit@{ki}                         : {h:.3f}")
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
