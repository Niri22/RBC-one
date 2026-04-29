"""VerifierAgent — Pass 2.5: critically reviews the SQLGeneratorAgent's draft answer.

The verifier sees the SQL code AND the draft answer/explanation and decides
whether to CONFIRM or REVISE the answer. This teaches multi-agent critique patterns.

Unlike SQLGeneratorAgent (which uses CrewAI + tool-use to explore the chart), the
verifier makes a single direct VLM call — showing that multi-agent critique
does not always require a full orchestration framework.

Key teaching point
------------------
Two agents looking at the plan and data can disagree. When the second model
(the verifier) has explicit access to the first model's reasoning, it can catch
errors the first model missed — KPI misread, SQL logic mistakes, etc.
"""

import base64
import os
from pathlib import Path
from typing import Any, Optional, Tuple
import json
import sqlite3
import textwrap
from google import genai
from openai import OpenAI


from ..langfuse_integration.tracing import close_span, open_llm_span
from ..utils.json_strict import parse_strict


VERIFIER_REQUIRED_KEYS = ["verdict", "answer", "reasoning"]

# Maximum rows shown to the verifier to keep the prompt compact.
_MAX_RESULT_ROWS = 20

_VERIFIER_PROMPT = """\
You are a critical senior data quality and QA verifier. A SQL generator agent has already attempted to answer
the question below. Your job: look at the SQL code and result carefully and audit the work.

Question         : {question}
Question Type    : {question_type}

Inspection plan the agent was supposed to follow:
{plan_steps}

SQLGeneratorAgent's Draft SQL:
{sql}
 
SQL Execution Result (first {max_rows} rows):
{sql_result}
 
SQLGeneratorAgent's Draft Answer: {draft_answer}
SQLGeneratorAgent's Draft Explanation: {draft_explanation}

Examine the SQL code. Then decide:
  CONFIRM — the draft answer is correct (output the same answer unchanged)
  REVISE  — you can see a clear, specific error; output the corrected answer

Rules:
- Only REVISE when you are confident you can point to a concrete error in the code
- If uncertain, CONFIRM — do not second-guess without confirming KPI definition 
- For MCQ questions: the answer must be one of the stated choices
- If the answer is truly unanswerable based on the data and KPI definition, say exactly "UNANSWERABLE"
- Keep answers concise — numbers, short phrases, or single words where appropriate
- Review the answer against baseline value

Output ONLY JSON, no markdown, no extra text:
{{"verdict": "confirmed" | "revised", "answer": "<final answer>", "reasoning": "<one sentence summerize the SQL query and calculation steps>"}}"""


# ---------------------------------------------------------------------------
# SQL execution helper
# ---------------------------------------------------------------------------
 
 
def _execute_sql(
    sql: str,
    data_source: Union["pd.DataFrame", sqlite3.Connection, None],  # noqa: F821
    max_rows: int = _MAX_RESULT_ROWS,
) -> str:
    """
    Run *sql* against *data_source* and return a compact string representation
    of the result suitable for inclusion in a prompt.
 
    Parameters
    ----------
    sql : str
        The SQL query to execute.
    data_source : pd.DataFrame | sqlite3.Connection | None
        - ``pd.DataFrame``: registered as the table ``"data"`` inside an
          in-process DuckDB connection, so the SQL should reference that name.
        - ``sqlite3.Connection``: executed directly.
        - ``None``: returns an informational message instead of raising.
 
    max_rows : int
        Maximum number of result rows to include in the returned string.
 
    Returns
    -------
    str
        A text table of results, an error message, or a not-available notice.
    """
    if data_source is None:
        return "(no data source provided — cannot execute SQL)"
 
    try:
        # ── pandas DataFrame via DuckDB ──────────────────────────────────────
        try:
            import duckdb  # optional fast path
            import pandas as pd
 
            if isinstance(data_source, pd.DataFrame):
                con = duckdb.connect()
                con.register("data", data_source)
                result_df = con.execute(sql).df()
                con.close()
                rows = result_df.head(max_rows)
                return rows.to_string(index=False)
        except ImportError:
            # DuckDB not available — fall through to sqlite3 path if applicable
            pass
 
        # ── sqlite3 Connection ───────────────────────────────────────────────
        if isinstance(data_source, sqlite3.Connection):
            cursor = data_source.execute(sql)
            col_names = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(max_rows)
            if not rows:
                return "(query returned no rows)"
            header = " | ".join(col_names)
            divider = "-" * len(header)
            body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
            return f"{header}\n{divider}\n{body}"
 
        return f"(unsupported data source type: {type(data_source).__name__})"
 
    except Exception as exc:  # noqa: BLE001
        return f"SQL ERROR: {exc}"
 
 
# ---------------------------------------------------------------------------
# Text-only LLM helpers
# ---------------------------------------------------------------------------
 
 
def _call_llm_openai(prompt: str, model: str, api_key: Optional[str]) -> str:
    """
    Submit a text completion request to the OpenAI API.
 
    Parameters
    ----------
    prompt : str
        The full verifier prompt.
    model : str
        The OpenAI model name (e.g. ``"gpt-4o-mini"``).
    api_key : str, optional
        API key; falls back to the ``OPENAI_API_KEY`` environment variable.
 
    Returns
    -------
    str
        Raw text response from the model.
    """
    from openai import OpenAI  # imported lazily — not required at module load
 
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=256,
        temperature=0,
    )
    return response.choices[0].message.content or ""
 
 
def _call_llm_gemini(prompt: str, model: str, api_key: Optional[str]) -> str:
    """
    Submit a text completion request to the Google Gemini API.
 
    Parameters
    ----------
    prompt : str
        The full verifier prompt.
    model : str
        The Gemini model name (e.g. ``"gemini-2.0-flash-lite"``).
    api_key : str, optional
        API key; falls back to the ``GEMINI_API_KEY`` environment variable.
 
    Returns
    -------
    str
        Raw text response from the model.
    """
    from google import genai  # imported lazily — not required at module load
 
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(temperature=0, max_output_tokens=256),
    )
    return response.text or ""
 
 
# ---------------------------------------------------------------------------
# VerifierAgent
# ---------------------------------------------------------------------------
 
 
class VerifierAgent:
    """
    A validation agent that critiques draft SQL answers against live query results.
 
    Executes the draft SQL against the provided data source, then passes the
    query, result, plan, and draft answer to a text LLM for a second-opinion
    audit.  Returns a structured verdict of CONFIRM or REVISE.
    """
 
    def __init__(
        self,
        backend: str = "gemini",
        model: str = "gemini-2.0-flash-lite",
        api_key: Optional[str] = None,
    ):
        """
        Initialise the verifier with the specified text LLM backend.
 
        Parameters
        ----------
        backend : {'openai', 'gemini'}
            The LLM provider used for the audit call.
        model : str
            The model name to request.
        api_key : str, optional
            API key to use for calls; falls back to the relevant environment
            variable if omitted.
        """
        self.backend = backend
        self.model = model
        self.api_key = api_key
 
    def run(
        self,
        sample,  # PerceivedSample
        plan: dict,
        vision_parsed: dict,
        data_source: Union["pd.DataFrame", sqlite3.Connection, None] = None,  # noqa: F821
        lf_trace: Any = None,
    ) -> Tuple[str, dict, bool, str]:
        """
        Execute the draft SQL and critically audit the draft answer.
 
        Parameters
        ----------
        sample : PerceivedSample
            The source data sample containing ``question``, ``question_type``,
            and optionally ``sql`` with the draft query.
        plan : dict
            The inspection plan used by the previous agent.
        vision_parsed : dict
            The draft answer and explanation to audit.  Expected keys:
            ``"answer"``, ``"explanation"``, and optionally ``"sql"``.
        data_source : pd.DataFrame | sqlite3.Connection | None, optional
            Live data against which to execute the SQL.  When ``None`` the
            verifier falls back to confirming without execution evidence.
        lf_trace : Any, optional
            Langfuse tracing object for observability.
 
        Returns
        -------
        prompt : str
            The verifier prompt rendered for the model.
        parsed : dict
            The final audited response containing ``'verdict'``, ``'answer'``,
            and ``'reasoning'``.
        parse_error : bool
            ``True`` if the JSON result was malformed.
        raw_text : str
            The raw string response from the LLM.
        """
        plan_steps = plan.get("steps", [])
        steps_text = (
            "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan_steps)) or "  (none)"
        )
 
        draft_answer = vision_parsed.get("answer", "(none)")
        draft_explanation = vision_parsed.get("explanation", "(none)")
 
        # Prefer SQL stored on the parsed output; fall back to the sample attribute.
        sql = vision_parsed.get("sql") or getattr(sample, "sql", "") or "(no SQL provided)"
 
        question_type = getattr(
            getattr(sample, "question_type", None),
            "value",
            str(getattr(sample, "question_type", "standard")),
        )
 
        # ── Execute SQL and capture results ──────────────────────────────────
        sql_result = _execute_sql(sql, data_source, max_rows=_MAX_RESULT_ROWS)
 
        prompt = _VERIFIER_PROMPT.format(
            question=sample.question,
            question_type=question_type,
            plan_steps=steps_text,
            sql=textwrap.indent(sql, "  "),
            sql_result=textwrap.indent(sql_result, "  "),
            draft_answer=draft_answer,
            draft_explanation=draft_explanation,
            max_rows=_MAX_RESULT_ROWS,
        )
 
        span = open_llm_span(
            lf_trace,
            name="verifier",
            input_data={"prompt": prompt, "draft_answer": draft_answer, "sql": sql},
            model=self.model,
            metadata={"backend": self.backend},
        )
 
        try:
            if self.backend == "openai":
                raw = _call_llm_openai(prompt, self.model, self.api_key)
            elif self.backend == "gemini":
                raw = _call_llm_gemini(prompt, self.model, self.api_key)
            else:
                raise ValueError(f"Unknown backend: {self.backend!r}")
 
            parsed, parse_ok = parse_strict(raw, required_keys=VERIFIER_REQUIRED_KEYS)
            if not parsed:
                parsed = {
                    "verdict": "confirmed",
                    "answer": draft_answer,
                    "reasoning": f"Parse error — defaulting to confirm. Raw: {raw[:120]}",
                }
                parse_ok = False
 
            # Normalise verdict to known values.
            if parsed.get("verdict", "").lower() not in ("confirmed", "revised"):
                parsed["verdict"] = "confirmed"
 
            close_span(span, output=parsed)
            return prompt, parsed, not parse_ok, raw
 
        except Exception as exc:
            fallback = {
                "verdict": "confirmed",
                "answer": draft_answer,
                "reasoning": f"Verifier error: {exc}",
            }
            close_span(span, output=fallback, error=str(exc))
            return prompt, fallback, True, ""