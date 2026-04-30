import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd
from crewai import LLM, Agent, Crew, Task

from ..utils.json_strict import parse_strict
from ..langfuse_integration.tracing import close_span, open_llm_span


SQL_RETRIEVAL_PROMPT_PATH = Path(__file__).parent / "prompts" / "sql_retrieval.txt"

# Adjust parents[5] to get from data folder
KPI_REGISTRY_PATH = Path(__file__).parents[5] / "data" / "KPI_Registry.csv"


DEFAULT_TABLE_NAME = "credit_card_clients"

SQL_RETRIEVAL_REQUIRED_KEYS = [
    "kpi_name",
    "source_tables",
    "source_fields",
    "join_keys",
    "data_freshness",
    "metric_logic",
    "match_confidence",
    "retrieval_notes",
]


def _load_template() -> str:
    return SQL_RETRIEVAL_PROMPT_PATH.read_text()


def _load_kpi_registry() -> str:
    df = pd.read_csv(KPI_REGISTRY_PATH)

    required_cols = [
        "KPI Name",
        "Description",
        "Key Fields",
        "Metric Type",
        "version",
        "Logic",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"KPI Registry is missing required columns: {missing}")

    return df.to_json(orient="records", indent=2)


def build_sql_retrieval_prompt(
    question: str,
    planner_output: dict,
    table_name: str = DEFAULT_TABLE_NAME,
) -> str:
    template = _load_template()

    return template.format(
        question=question,
        planner_output=json.dumps(planner_output, indent=2),
        table_name=table_name,
        kpi_registry=_load_kpi_registry(),
    )


def _build_llm(backend: str, model: str, api_key: Optional[str]) -> LLM:
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

    raise ValueError(f"Unknown SQL retrieval backend: {backend!r}")


class SQLRetrievalAgent:
    """
    Schema retrieval step for the MEP pipeline.

    This agent maps the user question and Planner output to the KPI Registry.
    It does not generate SQL. It returns table, field, and metric logic metadata
    for the SQL Generator Agent.
    """

    def __init__(
        self,
        backend: str = "gemini",
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
        table_name: str = DEFAULT_TABLE_NAME,
    ):
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.table_name = table_name
        self._llm = _build_llm(backend, model, api_key)

    def run(
        self,
        question: str,
        planner_output: dict,
        lf_trace: Any = None,
    ) -> Tuple[str, dict, bool, str]:
        prompt = build_sql_retrieval_prompt(
            question=question,
            planner_output=planner_output,
            table_name=self.table_name,
        )

        span = open_llm_span(
            lf_trace,
            name="schema_retriever",
            input_data={
                "question": question,
                "planner_output": planner_output,
                "prompt": prompt,
            },
            model=self.model,
            metadata={"backend": self.backend},
        )

        agent = Agent(
            role="KPI Schema Retriever",
            goal=(
                "Map the question and planner output to the approved KPI Registry. "
                "Return only the relevant KPI name, source table, source fields, "
                "join keys, data freshness, and metric logic."
            ),
            backstory=(
                "You are a schema retrieval specialist for a KPI interpretability system. "
                "You prevent downstream SQL generation errors by grounding every query "
                "in approved KPI definitions and fields from the KPI Registry."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

        task = Task(
            description=prompt,
            expected_output=(
                "A strict JSON object with keys: kpi_name, source_tables, "
                "source_fields, join_keys, data_freshness, metric_logic, "
                "match_confidence, retrieval_notes."
            ),
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        raw_text: str = getattr(result, "raw", None) or str(result)

        parsed, parse_ok = parse_strict(
            raw_text,
            required_keys=SQL_RETRIEVAL_REQUIRED_KEYS,
        )

        if parsed:
            parsed["source_tables"] = parsed.get("source_tables") or [self.table_name]
            parsed["source_fields"] = parsed.get("source_fields") or []
            parsed["join_keys"] = parsed.get("join_keys") or []
            parsed["retrieval_notes"] = parsed.get("retrieval_notes") or []

        close_span(
            span,
            output={
                "kpi_name": parsed.get("kpi_name") if parsed else None,
                "source_tables": parsed.get("source_tables") if parsed else [],
                "source_fields": parsed.get("source_fields") if parsed else [],
                "parse_error": not parse_ok,
            },
        )

        return prompt, parsed, not parse_ok, raw_text