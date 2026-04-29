"""LLM-as-judge evaluator for MEP outputs.

Scores five rubric dimensions (1–5) using a text LLM. All dimensions are
SQL-specific: they evaluate query correctness, source citation, reproducibility,
absence of hallucination, and KPI definition alignment.
"""

import os
from typing import Optional

from ..utils.json_strict import parse_strict


_JUDGE_PROMPT = """\
You are a senior data-quality reviewer auditing the output of an automated SQL \
metrics assistant. Evaluate the response strictly on the five dimensions below.

--- INPUT ---
Question      : {question}
Expected value: {expected}
Predicted value: {predicted}
Verifier verdict: {verifier_verdict}

--- SQL EVIDENCE ---
SQL query     : {sql}
Source tables : {source_tables}
Source fields : {source_fields}

--- SCORING RUBRIC ---
Score each dimension from 1 (poor) to 5 (excellent).

1. correctness
   Does the predicted value match the expected metric?
   5 = exact match or within acceptable numeric tolerance
   3 = directionally correct but imprecise
   1 = wrong value or wrong units

2. source_cited
   Are the source tables and fields explicitly listed and correct?
   5 = all relevant tables and fields named and appropriate
   3 = tables cited but fields vague or partially missing
   1 = no citations or wrong tables referenced

3. reproducibility
   Could another analyst re-run this SQL on the same DB and get the same answer?
   5 = query is deterministic, complete, and references no ambiguous aliases
   3 = query would likely reproduce but has minor ambiguity
   1 = query is missing clauses, uses SELECT *, or references unknown tables

4. no_hallucination
   Does the answer avoid stating assumptions, computed sub-values, or \
contextual details that are not derivable from the SQL or schema?
   5 = every claim is directly supported by the query result
   3 = minor unsupported claim present
   1 = answer invents data or attributes not in the result set

5. kpi_alignment
   Does the SQL metric logic match the business definition of the KPI implied \
by the question?
   5 = correct aggregation, filters, date range, and granularity
   3 = right metric family but wrong aggregation or filter
   1 = wrong metric computed entirely

Output ONLY valid JSON, no markdown fences, no extra text:
{{
  "correctness":      <1-5>,
  "source_cited":     <1-5>,
  "reproducibility":  <1-5>,
  "no_hallucination": <1-5>,
  "kpi_alignment":    <1-5>
}}
"""

_JUDGE_KEYS = [
    "correctness",
    "source_cited",
    "reproducibility",
    "no_hallucination",
    "kpi_alignment",
]


def _default_scores() -> dict:
    return dict.fromkeys(_JUDGE_KEYS, 0.0)


def _call_llm(prompt: str, backend: str, model: str, api_key: Optional[str]) -> str:
    """Send the judging prompt to the specified LLM backend.

    Parameters
    ----------
    backend : {'openai', 'gemini', 'anthropic'}
    """
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=512,
        )
        return resp.choices[0].message.content or ""

    if backend == "gemini":
        from google import genai
        client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.text or ""

    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""))
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text or ""

    raise ValueError(f"Unknown judge backend: {backend!r}")


def judge_mep(
    mep: dict,
    backend: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
) -> dict:
    """Score the SQL quality of a single MEP on five rubric dimensions (1–5).

    Evaluates correctness, citation accuracy, reproducibility, hallucination
    rate, and KPI alignment. Returns raw 1–5 scores per dimension.

    Parameters
    ----------
    mep : dict
        Deserialized MEP JSON.
    backend : {'openai', 'gemini', 'anthropic'}
        LLM provider for the judge.
    model : str
        Model name for the chosen backend.
    api_key : str, optional
        Provider API key. Falls back to environment variable.

    Returns
    -------
    dict
        Keys: correctness, source_cited, reproducibility, no_hallucination,
        kpi_alignment. Values are floats 1.0–5.0, or 0.0 on parse failure.
    """
    sample = mep.get("sample", {})
    sql = mep.get("sql_generator", {})
    verifier = mep.get("verifier") or {}

    question = sample.get("question", "")
    expected = sample.get("expected_output", "")
    predicted = (
        (verifier.get("parsed") or {}).get("answer")
        or sql.get("parsed", {}).get("answer", "")
    )
    verifier_verdict = verifier.get("verdict", "skipped")
    sql_query = sql.get("sql", "") or sql.get("parsed", {}).get("sql", "")
    source_tables = ", ".join(sql.get("source_tables", [])) or "none cited"
    source_fields = ", ".join(sql.get("source_fields", [])) or "none cited"

    prompt = _JUDGE_PROMPT.format(
        question=question,
        expected=expected,
        predicted=predicted,
        verifier_verdict=verifier_verdict,
        sql=sql_query,
        source_tables=source_tables,
        source_fields=source_fields,
    )

    try:
        raw = _call_llm(prompt, backend, model, api_key)
        scores, ok = parse_strict(raw, required_keys=_JUDGE_KEYS)
        if not scores:
            s = _default_scores()
            s["judge_parse_error"] = True
            return s
        # Normalise to float — LLM may return int
        return {k: float(scores.get(k, 0)) for k in _JUDGE_KEYS}
    except Exception as exc:
        s = _default_scores()
        s["judge_error"] = str(exc)
        return s
