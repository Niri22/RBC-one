"""SQLGeneratorAgent — CrewAI agent that converts a planner JSON plan into SQL.

Receives the structured plan from PlannerAgent and schema context from
SchemaRetrieverTool. Uses NL2SQLTool to generate and execute the query,
then extracts source tables, fields, and the metric answer for the MEP.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from crewai import LLM, Agent, Crew, Task
from crewai_tools import NL2SQLTool

from ..datasets.perceived_sample import PerceivedSample
from ..langfuse_integration.tracing import close_span, open_llm_span
from ..mep.schema import MEPSchemaRetriever
from ..utils.json_strict import parse_strict


SQL_GENERATOR_PROMPT_PATH = Path(__file__).parent / "prompts" / "sql_generator.txt"

SQL_REQUIRED_KEYS = [
    "sql",
    "answer",
    "explanation",
    "source_tables",
    "source_fields",
    "data_freshness",
    "guardrail_triggered",
    "fallback_used",
]

# ── Guardrail patterns ────────────────────────────────────────────────────────

_BLOCKED_PATTERNS = [
    re.compile(r"\bSELECT\s+\*", re.IGNORECASE),          # no SELECT *
    re.compile(r"\bDROP\b", re.IGNORECASE),                # no destructive ops
    re.compile(r"\bDELETE\b", re.IGNORECASE),
    re.compile(r"\bUPDATE\b", re.IGNORECASE),
    re.compile(r"\bINSERT\b", re.IGNORECASE),
    re.compile(r"\bCROSS\s+JOIN\b", re.IGNORECASE),        # no unbounded joins
]


def _load_template() -> str:
    return SQL_GENERATOR_PROMPT_PATH.read_text()


def _apply_guardrails(sql: str, allowed_tables: List[str]) -> Tuple[bool, str]:
    """
    Check SQL against guardrail rules.

    Parameters
    ----------
    sql : str
        The generated SQL string.
    allowed_tables : list of str
        Tables declared in the schema context. Any table not in this list
        triggers a guardrail violation.

    Returns
    -------
    triggered : bool
        True if any guardrail was violated.
    reason : str
        Human-readable explanation of the violation (empty if clean).
    """
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(sql):
            return True, f"Blocked pattern matched: {pattern.pattern}"

    if allowed_tables:
        sql_upper = sql.upper()
        for table in allowed_tables:
            # Remove table references found in the SQL from the check
            sql_upper = sql_upper.replace(table.upper(), "")
        # Rough heuristic: if unknown table names appear after FROM/JOIN
        from_join = re.findall(
            r"(?:FROM|JOIN)\s+(\w+)", sql, re.IGNORECASE
        )
        for ref in from_join:
            if ref.upper() not in [t.upper() for t in allowed_tables]:
                return True, f"Unknown table reference: {ref!r}"

    return False, ""


def build_sql_generator_prompt(
    sample: PerceivedSample,
    plan: Dict[str, Any],
    schema: Optional[MEPSchemaRetriever],
) -> str:
    """
    Render the SQL generator prompt from the plan and schema context.

    Parameters
    ----------
    sample : PerceivedSample
        Original question and metadata.
    plan : dict
        Parsed plan output from PlannerAgent (steps, expected_answer_type, etc.).
    schema : MEPSchemaRetriever or None
        Schema context from SchemaRetrieverTool. If None, agent works from
        plan hints only.

    Returns
    -------
    str
        Rendered prompt string.
    """
    template = _load_template()

    steps_text = "\n".join(
        f"  {i + 1}. {s}" for i, s in enumerate(plan.get("steps", []))
    )

    if schema:
        schema_block = (
            f"Source tables : {', '.join(schema.source_tables)}\n"
            f"Source fields : {', '.join(schema.source_fields)}\n"
            f"Join keys     : {', '.join(schema.join_keys)}\n"
            f"Data freshness: {schema.data_freshness}"
        )
    else:
        schema_block = "No schema context provided — infer from KPI definition."

    return template.format(
        question=sample.question,
        question_type=sample.question_type.value,
        expected_answer_type=plan.get("expected_answer_type", ""),
        answerability_check=plan.get("answerability_check", ""),
        hints="\n".join(f"  - {h}" for h in plan.get("hints", [])),
        plan_steps=steps_text,
        schema_block=schema_block,
        kpi_name=plan.get("kpi_name", "unknown"),
    )


def _build_llm(backend: str, model: str, api_key: Optional[str]) -> LLM:
    """Mirror PlannerAgent's _build_llm — same backends supported."""
    if backend == "openai":
        return LLM(
            model=model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            temperature=0,
        )
    if backend == "gemini":
        return LLM(
            model=f"gemini/{model}",
            api_key=api_key or os.environ.get("GEMINI_API_KEY", ""),
            temperature=0,
        )
    if backend == "anthropic":
        return LLM(
            model=f"anthropic/{model}",
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            temperature=0,
        )
    raise ValueError(f"Unknown SQL generator backend: {backend!r}")


class SQLGeneratorAgent:
    """
    CrewAI agent that converts a planner JSON plan into an executable SQL query.

    Wraps NL2SQLTool for query execution and enforces guardrails before
    writing results to the MEP. Mirrors PlannerAgent's interface so the
    Floater's pipeline wiring stays consistent.
    """

    def __init__(
        self,
        db_uri: str,
        backend: str = "anthropic",
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        db_uri : str
            SQLAlchemy-compatible database URI, e.g.
            ``"sqlite:///rbc_metrics.db"`` or
            ``"postgresql://user:pass@host/dbname"``.
            Read-only by default (NL2SQLTool default).
        backend : str
            LLM provider: 'openai', 'gemini', or 'anthropic'.
        model : str
            Model name for the chosen backend.
        api_key : str, optional
            Provider API key. Falls back to environment variable.
        """
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self._llm = _build_llm(backend, model, api_key)
        # NL2SQLTool is read-only by default — safe for production
        self._nl2sql = NL2SQLTool(db_uri=db_uri)

    def run(
        self,
        sample: PerceivedSample,
        plan: Dict[str, Any],
        schema: Optional[MEPSchemaRetriever] = None,
        lf_trace: Any = None,
    ) -> Tuple[str, dict, bool, str]:
        """
        Generate and execute SQL for the given plan.

        Parameters
        ----------
        sample : PerceivedSample
            Original question sample.
        plan : dict
            Parsed plan from PlannerAgent.run().
        schema : MEPSchemaRetriever, optional
            Schema context. If provided, used for guardrail table validation.
        lf_trace : Any, optional
            Langfuse trace object for observability.

        Returns
        -------
        prompt : str
            The rendered prompt sent to the agent.
        parsed : dict
            Structured output: sql, answer, explanation, source_tables,
            source_fields, data_freshness, guardrail_triggered, fallback_used.
        parse_error : bool
            True if JSON parsing failed.
        raw_text : str
            Raw LLM response.
        """
        prompt = build_sql_generator_prompt(sample, plan, schema)
        allowed_tables = schema.source_tables if schema else []

        span = open_llm_span(
            lf_trace,
            name="sql_generator",
            input_data={"prompt": prompt, "kpi": plan.get("kpi_name", "")},
            model=self.model,
            metadata={"backend": self.backend},
        )

        agent = Agent(
            role="SQL Metrics Generator",
            goal=(
                "Generate a precise, auditable SQL query that answers the KPI question. "
                "Always cite source tables and fields. Output JSON only — no extra text."
            ),
            backstory=(
                "You are a senior data engineer at RBC. You write exact, reproducible SQL "
                "queries for business metrics. You never use SELECT *, never reference "
                "tables outside the schema context, and always explain your metric logic "
                "in plain English so executives and analysts can verify the result."
            ),
            llm=self._llm,
            tools=[self._nl2sql],   # NL2SQLTool handles query execution
            verbose=False,
            allow_delegation=False,
        )

        task = Task(
            description=prompt,
            expected_output=(
                "A JSON object with keys: sql (string), answer (string), "
                "explanation (string), source_tables (list), source_fields (list), "
                "data_freshness (string), guardrail_triggered (bool), fallback_used (bool)"
            ),
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        raw_text: str = getattr(result, "raw", None) or str(result)
        parsed, parse_ok = parse_strict(raw_text, required_keys=SQL_REQUIRED_KEYS)

        if parsed:
            parsed = self._post_process(parsed, allowed_tables)

        close_span(
            span,
            output={
                "sql": parsed.get("sql", "") if parsed else "",
                "source_tables": parsed.get("source_tables", []) if parsed else [],
                "guardrail_triggered": parsed.get("guardrail_triggered", False) if parsed else False,
                "parse_error": not parse_ok,
            },
        )

        return prompt, parsed or {}, not parse_ok, raw_text

    def _post_process(self, parsed: dict, allowed_tables: List[str]) -> dict:
        """
        Apply guardrails and enforce citation requirement after parsing.

        Mutates and returns the parsed dict. Sets guardrail_triggered=True
        and clears sql if a violation is found. Sets fallback_used=True if
        source_tables is empty (incomplete data window).
        """
        sql = parsed.get("sql", "")

        # Guardrail check
        triggered, reason = _apply_guardrails(sql, allowed_tables)
        if triggered:
            parsed["guardrail_triggered"] = True
            parsed["sql"] = ""           # strip unsafe SQL before it reaches MEP
            parsed["answer"] = "GUARDRAIL_VIOLATION"
            parsed["explanation"] = f"Query blocked: {reason}"

        # Citation enforcement — source_tables must be non-empty
        if not parsed.get("source_tables"):
            parsed["fallback_used"] = True
            parsed["answer"] = parsed.get("answer", "UNANSWERABLE")

        # Normalise types
        for list_field in ("source_tables", "source_fields"):
            val = parsed.get(list_field, [])
            if isinstance(val, str):
                parsed[list_field] = [v.strip() for v in val.split(",") if v.strip()]

        return parsed