"""LLM-as-judge evaluator for MEP outputs.

Scores five rubric dimensions (0.0–1.0) using a text LLM.
"""

import os
from typing import Optional

from google import genai
from openai import OpenAI

from ..utils.json_strict import parse_strict


_JUDGE_PROMPT = """\
You are evaluating the output of an internal metrics assistant.

Question: {question}
Expected metric: {expected}
Predicted answer: {predicted}
SQL query used: {sql}
Source tables cited: {source_tables}

Score each dimension 1-5. Output ONLY JSON, no markdown, no extra text::
{{
  "correctness":      <1-5>,  // Does the answer match the expected metric?
  "source_cited":     <1-5>,  // Are source tables and fields clearly identified?
  "reproducibility":  <1-5>,  // Could this SQL be re-run to get the same result?
  "no_hallucination": <1-5>,  // Does the answer avoid presenting assumptions as facts?
  "kpi_alignment":    <1-5>   // Does the answer match the KPI definition for this metric?
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
    """
    Generate a baseline scores dictionary with all metrics set to zero.

    Returns
    -------
    dict
        The initialized scores mapping.
    """
    return dict.fromkeys(_JUDGE_KEYS, 0.0)


def _call_llm(prompt: str, backend: str, model: str, api_key: Optional[str]) -> str:
    """
    Send a judging prompt to the specified backend.

    Parameters
    ----------
    prompt : str
        The evaluation rubric and data.
    backend : {'openai', 'gemini'}
        The model provider.
    model : str
        The specific model name.
    api_key : str, optional
        Provider API key.

    Returns
    -------
    str
        The model's textual assessment.
    """
    if backend == "openai":
        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=512,
        )
        return resp.choices[0].message.content or ""

    if backend == "gemini":
        client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.text or ""

    raise ValueError(f"Unknown judge backend: {backend!r}")


def judge_mep(
    mep: dict,
    backend: str = "gemini",
    model: str = "gemini-2.5-flash-lite",
    api_key: Optional[str] = None,
) -> dict:
    """
    Score the quality of a single agent execution record.

    Uses an LLM to evaluate faithfulness, plan adherence, and groundedness.

    Parameters
    ----------
    mep : dict
        The execution trace to judge.
    backend : str, default 'gemini'
        The provider for the judge model.
    model : str, default 'gemini-2.5-flash-lite'
        The model name.
    api_key : str, optional
        API key for the judge.

    Returns
    -------
    dict
        A dictionary of numeric scores and qualitative reasoning.
    """
    sample  = mep.get("sample", {})
    sql     = mep.get("sql_generator", {})              # was vision
    verifier = mep.get("verifier") or {}

    question  = sample.get("question", "")
    expected  = sample.get("expected_output", "")
    # Prefer verifier answer, fall back to sql_generator
    predicted = (verifier.get("parsed") or {}).get("answer") \
                or sql.get("parsed", {}).get("answer", "")
    sql_query       = sql.get("sql", "")
    source_tables   = ", ".join(sql.get("source_tables", [])) or "none cited"

    prompt = _JUDGE_PROMPT.format(
        question=question,
        expected=expected,        # matches {expected} in prompt template
        predicted=predicted,      # matches {predicted} in prompt template
        sql=sql_query,
        source_tables=source_tables,
    )

    try:
        raw = _call_llm(prompt, backend, model, api_key)
        scores, ok = parse_strict(raw, required_keys=_JUDGE_KEYS)
        if not scores:
            scores = _default_scores()
            scores["judge_parse_error"] = True
        return scores
    except Exception as exc:
        s = _default_scores()
        s["judge_error"] = str(exc)
        return s
