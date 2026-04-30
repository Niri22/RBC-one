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
from typing import Any, Optional, Tuple, Union
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
You are a senior SQL and data quality auditor. A SQL generator agent has already attempted to answer
the question below by writing and executing SQL. Your job is to audit that SQL and decide whether the
draft answer is correct.

Question           : {question}
Question Type      : {question_type}
Expected Answer Type: {expected_answer_type}

Inspection plan the agent was supposed to follow:
{plan_steps}

Hints (from planner):
{hints}

SQLGeneratorAgent's Draft SQL:
{sql}

SQL Execution Result (first {max_rows} rows):
{sql_result}

Draft Answer     : {draft_answer}
Draft Explanation: {draft_explanation}
Source Tables    : {source_tables}
Source Fields    : {source_fields}
Data Freshness   : {data_freshness}
Fallback Used    : {fallback_used}

AUDIT CHECKLIST — work through each point before deciding:
1. Aggregation function: Is AVG/SUM/COUNT the correct one for this KPI? For a rate on a 0/1 column, AVG is correct; COUNT(*) is wrong.
2. CAST / numeric scale: If AVG is applied to an integer 0/1 column, confirm CAST(...AS FLOAT) or CAST(...AS REAL) is present. A bare AVG on INTEGER in SQLite returns an integer.
3. Percentage format: If the expected answer type is a percentage (e.g. "22.12%"), confirm the SQL multiplies by 100 and rounds. If not, the answer will be off by 100×.
4. Filter values: Verify WHERE clause uses the correct coded values for categorical columns (e.g. SEX: 1=male, 2=female; EDUCATION: 1=grad, 2=university, 3=high school, 4=other; MARRIAGE: 1=married, 2=single, 3=other). A filter like SEX=0 returns 0 rows.
5. SQL result vs draft answer: Does the first row of the SQL result match the draft answer (within rounding)? If the result says 22.12 but the answer says "0.2212", that is a scale error.
6. Empty result: If the SQL result says "(query returned no rows)" but the draft answer is a number, that is a definite error — REVISE.
7. MCQ alignment: For multiple-choice questions, the answer must be exactly one of the stated choices.

Decide:
  CONFIRM — the draft answer is consistent with the SQL result and the audit checklist passes
  REVISE  — you found a concrete, specific error above; output the corrected answer derived from the SQL result

Rules:
- REVISE only when you can point to a specific item in the audit checklist that fails
- If the SQL result contradicts the draft answer (different number, wrong scale, 0 rows), always REVISE
- If the answer is truly unanswerable from the available data, output exactly "UNANSWERABLE"
- Keep answers concise — numbers (with % if a rate), short phrases, or single words where appropriate
- Do not invent data not present in the SQL result

Output ONLY JSON, no markdown, no extra text:
{{"verdict": "confirmed" | "revised", "answer": "<final answer>", "reasoning": "<one sentence: what the SQL computes and whether/why you changed the answer>"}}"""



# ---------------------------------------------------------------------------
# SQL execution helper
# ---------------------------------------------------------------------------


def _execute_sql(
    sql: str,
    data_source: Union[str, sqlite3.Connection, None],
    max_rows: int = _MAX_RESULT_ROWS,
) -> str:
    """
    Run *sql* against *data_source* and return a compact string representation
    of the result suitable for inclusion in a prompt.

    Parameters
    ----------
    sql : str
        The SQL query to execute.  An empty string or sentinel value returns
        an informational notice without raising.
    data_source : str | sqlite3.Connection | None
        - ``str``: treated as a SQLite file path (e.g. from a
          ``"sqlite:///path/to/file.db"`` URI — the ``sqlite:///`` prefix is
          stripped automatically).
        - ``sqlite3.Connection``: used directly; useful in tests.
        - ``None``: returns an informational message instead of raising.

    max_rows : int
        Maximum number of result rows to include in the returned string.

    Returns
    -------
    str
        A text table of results, an error message, or a not-available notice.
    """
    if not sql or sql.strip() == "(no SQL provided)":
        return "(no SQL to execute)"

    if data_source is None:
        return "(no data source provided — cannot execute SQL)"

    try:
        # ── Resolve a URI string to a sqlite3 connection ─────────────────────
        if isinstance(data_source, str):
            # Strip SQLAlchemy sqlite prefix so sqlite3 can open the file.
            path = data_source
            for prefix in ("sqlite:///", "sqlite://"):
                if path.startswith(prefix):
                    path = path[len(prefix):]
                    break
            conn = sqlite3.connect(path)
            close_conn = True
        elif isinstance(data_source, sqlite3.Connection):
            conn = data_source
            close_conn = False
        else:
            return f"(unsupported data source type: {type(data_source).__name__})"

        try:
            cursor = conn.execute(sql)
            col_names = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(max_rows)
        finally:
            if close_conn:
                conn.close()

        if not rows:
            return "(query returned no rows)"

        header = " | ".join(col_names)
        divider = "-" * len(header)
        body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
        return f"{header}\n{divider}\n{body}"

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
        max_completion_tokens=512,
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
        config=genai.types.GenerateContentConfig(temperature=0, max_output_tokens=512),
    )
    return response.text or ""


def _call_llm_anthropic(prompt: str, model: str, api_key: Optional[str]) -> str:
    """
    Submit a text completion request to the Anthropic API.

    Parameters
    ----------
    prompt : str
        The full verifier prompt.
    model : str
        The Anthropic model name (e.g. ``"claude-sonnet-4-6"``).
    api_key : str, optional
        API key; falls back to the ``ANTHROPIC_API_KEY`` environment variable.

    Returns
    -------
    str
        Raw text response from the model.
    """
    import anthropic  # imported lazily — not required at module load

    client = anthropic.Anthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    )
    message = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text if message.content else ""


# ---------------------------------------------------------------------------
# VerifierAgent
# ---------------------------------------------------------------------------


class VerifierAgent:
    """
    A validation agent that critiques SQLGeneratorAgent's draft SQL answer.

    Connects to the same database used by the generator, re-executes the draft
    SQL to obtain live results, then passes the full generator output — SQL,
    execution result, source citations, and flags — to a text LLM for a
    second-opinion audit.  Returns a structured verdict of CONFIRM or REVISE.

    Short-circuit behaviour
    -----------------------
    If ``sql_parsed["guardrail_triggered"]`` is ``True`` the LLM call is skipped
    entirely: the verifier confirms the ``GUARDRAIL_VIOLATION`` answer unchanged.
    This avoids wasting tokens auditing SQL that has already been sanitised.

    Usage (from the Floater pipeline)
    -----------------------------------
    ::

        _, sql_parsed, sql_err, sql_raw = sql_generator.run(sample, plan, schema, lf_trace)

        _, ver_parsed, ver_err, ver_raw = verifier.run(
            sample=sample,
            plan=plan,
            sql_parsed=sql_parsed,
            lf_trace=lf_trace,
        )
        final_answer = ver_parsed["answer"]
    """

    def __init__(
        self,
        db_uri: str,
        backend: str = "anthropic",
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
    ):
        """
        Initialise the verifier with the same database as SQLGeneratorAgent.

        Parameters
        ----------
        db_uri : str
            SQLAlchemy-compatible SQLite URI, e.g. ``"sqlite:///rbc_metrics.db"``.
            Must match the ``db_uri`` passed to ``SQLGeneratorAgent`` so the
            verifier re-executes SQL against the same data.
        backend : {'openai', 'gemini', 'anthropic'}
            The LLM provider used for the audit call.
        model : str
            The model name to request.
        api_key : str, optional
            API key to use for calls; falls back to the relevant environment
            variable if omitted.
        """
        self.db_uri = db_uri
        self.backend = backend
        self.model = model
        self.api_key = api_key

    def run(
        self,
        sample,  # PerceivedSample
        plan: dict,
        sql_parsed: dict,
        lf_trace: Any = None,
    ) -> Tuple[str, dict, bool, str]:
        """
        Re-execute the draft SQL and critically audit SQLGeneratorAgent's answer.

        Parameters
        ----------
        sample : PerceivedSample
            The original question sample (provides ``question`` and
            ``question_type``).
        plan : dict
            The inspection plan from PlannerAgent (used for ``steps``).
        sql_parsed : dict
            The ``parsed`` dict returned by ``SQLGeneratorAgent.run()``.
            Expected keys (SQL_REQUIRED_KEYS):
            ``"sql"``, ``"answer"``, ``"explanation"``, ``"source_tables"``,
            ``"source_fields"``, ``"data_freshness"``,
            ``"guardrail_triggered"``, ``"fallback_used"``.
        lf_trace : Any, optional
            Langfuse tracing object for observability.

        Returns
        -------
        prompt : str
            The verifier prompt rendered for the model (empty string on
            guardrail short-circuit).
        parsed : dict
            The final audited response containing ``'verdict'``, ``'answer'``,
            and ``'reasoning'``.
        parse_error : bool
            ``True`` if the JSON result was malformed.
        raw_text : str
            The raw string response from the LLM (empty on short-circuit).
        """
        # ── Unpack SQLGeneratorAgent output ──────────────────────────────────
        sql               = sql_parsed.get("sql", "") or ""
        draft_answer      = sql_parsed.get("answer", "(none)")
        draft_explanation = sql_parsed.get("explanation", "(none)")
        source_tables     = sql_parsed.get("source_tables", [])
        source_fields     = sql_parsed.get("source_fields", [])
        data_freshness    = sql_parsed.get("data_freshness", "(unknown)")
        guardrail_triggered = bool(sql_parsed.get("guardrail_triggered", False))
        fallback_used     = bool(sql_parsed.get("fallback_used", False))

        # ── Short-circuit: guardrail already fired ────────────────────────────
        if guardrail_triggered:
            short_circuit = {
                "verdict": "confirmed",
                "answer": draft_answer,
                "reasoning": (
                    "Guardrail triggered by SQLGeneratorAgent — "
                    "SQL was sanitised; verifier confirms without re-execution."
                ),
            }
            span = open_llm_span(
                lf_trace,
                name="verifier",
                input_data={"guardrail_short_circuit": True, "draft_answer": draft_answer},
                model=self.model,
                metadata={"backend": self.backend, "guardrail_triggered": True},
            )
            close_span(span, output=short_circuit)
            return "", short_circuit, False, ""

        # ── Plan steps, expected answer type, and hints ──────────────────────
        plan_steps = plan.get("steps", [])
        steps_text = (
            "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan_steps))
            or "  (none)"
        )
        expected_answer_type = plan.get("expected_answer_type", "(unknown)")
        hints = plan.get("hints", [])
        hints_text = (
            "\n".join(f"  - {h}" for h in hints) if hints else "  (none)"
        )

        question_type = getattr(
            getattr(sample, "question_type", None),
            "value",
            str(getattr(sample, "question_type", "standard")),
        )

        # ── Re-execute SQL against the same db_uri ────────────────────────────
        sql_result = _execute_sql(sql, self.db_uri, max_rows=_MAX_RESULT_ROWS)

        # ── Render prompt ─────────────────────────────────────────────────────
        prompt = _VERIFIER_PROMPT.format(
            question=sample.question,
            question_type=question_type,
            expected_answer_type=expected_answer_type,
            plan_steps=steps_text,
            hints=hints_text,
            sql=textwrap.indent(sql or "(no SQL provided)", "  "),
            sql_result=textwrap.indent(sql_result, "  "),
            draft_answer=draft_answer,
            draft_explanation=draft_explanation,
            source_tables=", ".join(source_tables) if source_tables else "(none)",
            source_fields=", ".join(source_fields) if source_fields else "(none)",
            data_freshness=data_freshness,
            fallback_used=fallback_used,
            max_rows=_MAX_RESULT_ROWS,
        )

        span = open_llm_span(
            lf_trace,
            name="verifier",
            input_data={
                "prompt": prompt,
                "draft_answer": draft_answer,
                "sql": sql,
                "source_tables": source_tables,
                "fallback_used": fallback_used,
            },
            model=self.model,
            metadata={"backend": self.backend, "db_uri": self.db_uri},
        )

        try:
            if self.backend == "openai":
                raw = _call_llm_openai(prompt, self.model, self.api_key)
            elif self.backend == "gemini":
                raw = _call_llm_gemini(prompt, self.model, self.api_key)
            elif self.backend == "anthropic":
                raw = _call_llm_anthropic(prompt, self.model, self.api_key)
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
