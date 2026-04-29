# SQL Assistant — Agentic Evaluation Framework

## Overview

This document maps the **SQL Assistant pipeline** onto every file in this repository, identifies which files have been updated for the new SQL architecture, and lists what still needs work.

The SQL Assistant replaces the original ChartQAPro vision pipeline (Plan → OCR → VisionAgent) with a text-only SQL generation pipeline (Plan → SchemaRetriever → SQLGenerator → Verifier). The codebase is currently a **hybrid** — the SQL pipeline is the active implementation, but legacy ChartQAPro files are still present and need to be cleaned up or adapted.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Input Sample                        │
│          NL question · schema · expected metric          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                     PlannerAgent                         │
│         Text LLM — identifies KPI intent                 │
│         Outputs structured query plan (2–4 steps)        │
└──────────────────────────┬──────────────────────────────┘
                           │ plan.steps
                           ▼
┌─────────────────────────────────────────────────────────┐  ┐
│                  SchemaRetrieverTool                     │  │
│         Single call — retrieves table metadata           │  │ optional
│         Outputs schema context · field definitions       │  │
└──────────────────────────┬──────────────────────────────┘  ┘
                           │ schema context
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  SQLGeneratorAgent                       │
│         Text LLM + SQL generation tool (once)            │
│         Outputs SQL query · metric calculation           │
└──────────────────────────┬──────────────────────────────┘
                           │ draft SQL + explanation
                           ▼
┌─────────────────────────────────────────────────────────┐  ┐
│               VerifierAgent — Pass 2.5                   │  │
│         Re-checks SQL logic · KPI definitions            │  │ optional
│         Outputs: confirmed or revised · rationale        │  │
└──────────────────────────┬──────────────────────────────┘  ┘
                           │ final SQL
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Model Evaluation Packet (MEP)               │
│         JSON: plan · schema · sql · verifier             │
│         run_id · config · timestamps · errors            │
└──────┬─────────────────────┬──────────────────────┬─────┘
       │                     │                      │
       ▼                     ▼                      ▼
┌────────────┐       ┌─────────────┐       ┌──────────────┐
│eval_outputs│       │ eval_traces │       │  eval_topk   │
│Accuracy +  │       │Latency · SQL│       │ hit@1/hit@2/ │
│LLM judge   │       │calls · Replay│      │    hit@3     │
│5 rubric    │       │    score    │       │ Top-K re-    │
│ dimensions │       └──────┬──────┘       │  querying    │
└─────┬──────┘              │              └──────┬───────┘
      └────────────────────┬┘                     │
                           ▼
                  ┌────────────────┐
                  │  summarize.py  │
                  │  summary.csv   │
                  │by config×qtype │
                  └────────────────┘
```

---

## File Map — Architecture Stage by Stage

### Stage 0: Input Sample

| File | Role | Status |
|------|------|--------|
| [eval/eval_samples.json](src/agentic_chartqapro_eval/eval/eval_samples.json) | Evaluation dataset — `sample_id`, `question`, `question_type`, `expected_output`, `kpi_name` | ✅ Updated |
| [datasets/perceived_sample.py](src/agentic_chartqapro_eval/datasets/perceived_sample.py) | `PerceivedSample` dataclass — dataset-agnostic sample wrapper shared by all agents | ✅ Updated |

### Stage 1: PlannerAgent

| File | Role | Status |
|------|------|--------|
| [agents/planner_agent.py](src/agentic_chartqapro_eval/agents/planner_agent.py) | `PlannerAgent` — text-only LLM that reads the question and emits a 2–4 step query plan (`steps`, `expected_answer_type`, `answerability_check`, `hints`) | ✅ Updated |
| [agents/prompts/planner.txt](src/agentic_chartqapro_eval/agents/prompts/planner.txt) | Planner system prompt — instructs the LLM to identify KPI intent and break it into inspection steps | ✅ Updated |

### Stage 2: SchemaRetrieverTool (optional)

| File | Role | Status |
|------|------|--------|
| [agents/sql_retrieval_agent.py](src/agentic_chartqapro_eval/agents/sql_retrieval_agent.py) | `SQLRetrievalAgent` — maps question + planner output to KPI Registry, returns `kpi_name`, `source_tables`, `source_fields`, `join_keys`, `metric_logic` | ✅ Updated |
| [agents/prompts/sql_retrieval.txt](src/agentic_chartqapro_eval/agents/prompts/sql_retrieval.txt) | Schema retrieval prompt — instructs the LLM to match the user's question to an approved KPI definition from the registry | ✅ Updated |

### Stage 3: SQLGeneratorAgent

| File | Role | Status |
|------|------|--------|
| [agents/sqlgenerator_agent.py](src/agentic_chartqapro_eval/agents/sqlgenerator_agent.py) | `SQLGeneratorAgent` — CrewAI agent + NL2SQLTool. Enforces guardrails (blocks `SELECT *`, `DROP`, `CROSS JOIN`, unknown tables). Requires non-empty `source_tables`. Returns `sql`, `answer`, `explanation`, `guardrail_triggered`, `fallback_used` | ✅ Updated |
| [agents/prompts/sglgenerator.txt](src/agentic_chartqapro_eval/agents/prompts/sglgenerator.txt) | SQL generator system prompt — instructs LLM to produce auditable SQL with citations | ✅ Updated |

### Stage 4: VerifierAgent — Pass 2.5 (optional)

| File | Role | Status |
|------|------|--------|
| [agents/verifier_agent.py](src/agentic_chartqapro_eval/agents/verifier_agent.py) | `VerifierAgent` — single direct VLM call that re-examines draft SQL logic and KPI definitions. Returns `verdict` (`confirmed`\|`revised`), `answer`, `reasoning` | ✅ Updated |

### Stage 5: Model Evaluation Packet (MEP)

| File | Role | Status |
|------|------|--------|
| [mep/schema.py](src/agentic_chartqapro_eval/mep/schema.py) | MEP v2 dataclasses — `MEP`, `MEPConfig`, `MEPSample`, `MEPPlan`, `MEPSchemaRetriever`, `MEPSQLGenerator`, `MEPVerifier`, `MEPTimestamps`, `ToolTrace` | ✅ Updated |
| [mep/writer.py](src/agentic_chartqapro_eval/mep/writer.py) | MEP I/O — `write_mep()`, `read_mep()`, `iter_meps()`. Stub builder functions (`init_mep`, `append_plan`, etc.) not yet implemented | ⚠️ Needs work |

### Stage 6: Pipeline Runner

| File | Role | Status |
|------|------|--------|
| [runner/run_generate_meps.py](src/agentic_chartqapro_eval/runner/run_generate_meps.py) | Main CLI entry point — orchestrates Planner → SchemaRetriever → SQLGenerator → Verifier, writes MEPs to disk. Supports `--config`, `--csv`, `--db_uri`, `--n`, `--workers`, `--no_verifier`, `--no_schema_retriever` | ✅ Updated |
| [eval/eval_runner.py](src/agentic_chartqapro_eval/eval/eval_runner.py) | Alternative unified eval harness — similar orchestration to `run_generate_meps.py` | ⚠️ Partially duplicates runner; relationship to `run_generate_meps.py` unclear |
| [eval/db_setup.py](src/agentic_chartqapro_eval/eval/db_setup.py) | Loads UCI credit card CSV into SQLite for evaluation; returns SQLAlchemy URI | ✅ Updated |

### Stage 7: Evaluation Passes

| File | Evaluation Pass | Role | Status |
|------|----------------|------|--------|
| [eval/eval_outputs.py](src/agentic_chartqapro_eval/eval/eval_outputs.py) | Pass 1 — Accuracy + LLM judge | Scores answer accuracy, citation presence, guardrail hits, verifier verdict, 5-dimension judge rubric | ✅ Updated |
| [eval/eval_traces.py](src/agentic_chartqapro_eval/eval/eval_traces.py) | Pass 2 — Latency + replayability | Computes latency per stage, SQL tool call count, fallback rate, replayability score | ✅ Updated |
| [eval/eval_topk.py](src/agentic_chartqapro_eval/eval/eval_topk.py) | Pass 3 — Top-K hits | Re-queries for hit@1/hit@2/hit@3 candidates | ⚠️ Needs update |
| [eval/error_taxonomy.py](src/agentic_chartqapro_eval/eval/error_taxonomy.py) | Pass 4 — Failure taxonomy | VLM-based failure classification — categories still reference chart QA failures | ⚠️ Needs update |
| [eval/judge.py](src/agentic_chartqapro_eval/eval/judge.py) | LLM-as-judge | 5-dimension rubric scoring (`explanation_quality`, `hallucination_rate`, `plan_coverage`, `plan_adherence`, `faithfulness_alignment`) | ⚠️ Rubric assumes chart analysis context |
| [eval/summarize.py](src/agentic_chartqapro_eval/eval/summarize.py) | Aggregation | Rolls up metrics.jsonl → summary.csv by config × question_type | ✅ Updated |
| [eval/report.py](src/agentic_chartqapro_eval/eval/report.py) | HTML report | Generates summary cards and per-sample table — designed for chart QA output fields | ⚠️ Needs update |
| [eval/dashboard.py](src/agentic_chartqapro_eval/eval/dashboard.py) | Streamlit dashboard | Interactive sample browser — reads chart-QA-specific MEP fields | ⚠️ Needs update |

### Supporting Infrastructure

| File | Role | Status |
|------|------|--------|
| [langfuse_integration/client.py](src/agentic_chartqapro_eval/langfuse_integration/client.py) | Langfuse singleton — gracefully degrades if credentials absent | ✅ Active |
| [langfuse_integration/tracing.py](src/agentic_chartqapro_eval/langfuse_integration/tracing.py) | Trace span helpers — `sample_trace`, `open_llm_span`, `close_span`, `log_trace_scores` | ✅ Active |
| [langfuse_integration/prompts.py](src/agentic_chartqapro_eval/langfuse_integration/prompts.py) | Pushes prompt versions to Langfuse Prompt Management | ⚠️ Still registers `vision.txt`; needs SQL prompts |
| [langfuse_integration/dataset.py](src/agentic_chartqapro_eval/langfuse_integration/dataset.py) | Registers eval dataset in Langfuse | ⚠️ May still reference ChartQAPro dataset |
| [langfuse_integration/ingest.py](src/agentic_chartqapro_eval/langfuse_integration/ingest.py) | Retroactively imports MEP files into Langfuse | ✅ Active |
| [utils/json_strict.py](src/agentic_chartqapro_eval/utils/json_strict.py) | JSON parse with fence-stripping, block extraction, and repair fallback | ✅ Active |
| [utils/timing.py](src/agentic_chartqapro_eval/utils/timing.py) | `timed()` context manager, `iso_now()` helper | ✅ Active |
| [utils/hashing.py](src/agentic_chartqapro_eval/utils/hashing.py) | SHA256 image hashing — used in original ChartQAPro pipeline for image deduplication | ⚠️ Legacy |

---

## Legacy Files (ChartQAPro — Not Used in SQL Pipeline)

These files implement the original vision-based pipeline and are **not imported or called** by `run_generate_meps.py`. They should either be removed or clearly namespaced as legacy examples.

| File | Original Role | Replacement in SQL Pipeline |
|------|--------------|----------------------------|
| [agents/vision_agent.py](src/agentic_chartqapro_eval/agents/vision_agent.py) | VisionAgent — CrewAI agent that calls VisionQATool once to analyze chart image | `sqlgenerator_agent.py` |
| [tools/vision_qa_tool.py](src/agentic_chartqapro_eval/tools/vision_qa_tool.py) | VisionQATool — multimodal VLM for chart image understanding | `sqlgenerator_agent.py` (via NL2SQLTool) |
| [tools/ocr_reader_tool.py](src/agentic_chartqapro_eval/tools/ocr_reader_tool.py) | OcrReaderTool — extracts all visible text from chart image | `sql_retrieval_agent.py` |
| [agents/prompts/vision.txt](src/agentic_chartqapro_eval/agents/prompts/vision.txt) | Chart analysis prompt for VisionAgent | `sglgenerator.txt` |
| [datasets/chartqapro_loader.py](src/agentic_chartqapro_eval/datasets/chartqapro_loader.py) | Loads ChartQAPro dataset from HuggingFace, saves chart images locally | `eval_samples.json` (flat JSON, no HuggingFace) |
| [README.md](README.md) | ChartQAPro pipeline documentation — Plan → OCR → VisionAgent flow | This file |

---

## To-Do List

### High Priority — Blocks Correct Evaluation

- [ ] **[eval/eval_topk.py](src/agentic_chartqapro_eval/eval/eval_topk.py)** — Rewrite top-K pass for SQL pipeline. Current implementation re-queries a VLM with chart images. New version should re-run `SQLGeneratorAgent` K times (or with temperature > 0) and compute hit@1/hit@2/hit@3 against `expected_output`.

- [ ] **[eval/judge.py](src/agentic_chartqapro_eval/eval/judge.py)** — Update the 5-dimension judge rubric for SQL context. Dimensions like `hallucination_rate` and `faithfulness_alignment` still assume the judge is reading a chart description. Update prompts to evaluate SQL correctness, citation accuracy, and metric logic fidelity instead.

- [ ] **[mep/writer.py](src/agentic_chartqapro_eval/mep/writer.py)** — Implement the stub builder functions (`init_mep`, `append_plan`, `append_schema`, `append_sql`, `append_verifier`, `close_mep`). These are called by the runner but currently no-ops, meaning in-flight MEPs are lost if the pipeline crashes mid-sample.

### Medium Priority — Correctness and Clean-Up

- [ ] **[eval/error_taxonomy.py](src/agentic_chartqapro_eval/eval/error_taxonomy.py)** — Replace chart-QA failure categories (`axis_misread`, `legend_confusion`, `hallucinated_element`) with SQL-appropriate taxonomy: `wrong_table`, `wrong_aggregation`, `wrong_filter`, `date_range_error`, `join_error`, `metric_definition_mismatch`, `guardrail_blocked`, `parse_failure`.

- [ ] **[eval/report.py](src/agentic_chartqapro_eval/eval/report.py)** — Update HTML report to display SQL-specific fields: generated SQL query, source tables, citation status, guardrail hit, verifier verdict. Remove chart image rendering and OCR output sections.

- [ ] **[eval/dashboard.py](src/agentic_chartqapro_eval/eval/dashboard.py)** — Update Streamlit dashboard to read `mep.sql_generator` and `mep.schema_retriever` instead of `mep.vision` and `mep.ocr`. Add SQL query syntax highlighting, table lineage view, and verifier reasoning panel.

- [ ] **[eval/eval_runner.py](src/agentic_chartqapro_eval/eval/eval_runner.py)** — Clarify the relationship with `runner/run_generate_meps.py`. These two files overlap significantly. Either consolidate into one entry point, or clearly separate responsibilities (e.g., `run_generate_meps.py` for batch generation, `eval_runner.py` as a programmatic API).

- [ ] **[langfuse_integration/prompts.py](src/agentic_chartqapro_eval/langfuse_integration/prompts.py)** — Register the three active SQL prompts (`planner.txt`, `sql_retrieval.txt`, `sglgenerator.txt`) in Langfuse Prompt Management. Remove or deprecate the `vision.txt` registration.

- [ ] **[langfuse_integration/dataset.py](src/agentic_chartqapro_eval/langfuse_integration/dataset.py)** — Point dataset registration at `eval_samples.json` instead of the ChartQAPro HuggingFace dataset.

### Low Priority — Housekeeping

- [ ] **[agents/vision_agent.py](src/agentic_chartqapro_eval/agents/vision_agent.py)** — Move to a `legacy/` subfolder or delete. It is not imported anywhere in the SQL pipeline and creates confusion about what agents are active.

- [ ] **[tools/vision_qa_tool.py](src/agentic_chartqapro_eval/tools/vision_qa_tool.py)** — Move to `legacy/` or delete alongside `vision_agent.py`.

- [ ] **[tools/ocr_reader_tool.py](src/agentic_chartqapro_eval/tools/ocr_reader_tool.py)** — Move to `legacy/` or delete alongside `vision_agent.py`.

- [ ] **[agents/prompts/vision.txt](src/agentic_chartqapro_eval/agents/prompts/vision.txt)** — Move to `legacy/` or delete.

- [ ] **[datasets/chartqapro_loader.py](src/agentic_chartqapro_eval/datasets/chartqapro_loader.py)** — Move to `legacy/` or delete. SQL pipeline loads samples from `eval_samples.json` directly.

- [ ] **[utils/hashing.py](src/agentic_chartqapro_eval/utils/hashing.py)** — Verify whether this is still used anywhere in the SQL pipeline. If not, delete or move to `legacy/`.

- [ ] **[README.md](README.md)** — The existing README documents the ChartQAPro pipeline. Update it (or replace it with a pointer to this file) once the SQL pipeline is stable.

---

## Architecture Alignment Summary

| Architecture Stage | File(s) | Updated? |
|-------------------|---------|----------|
| Input sample | `eval_samples.json`, `perceived_sample.py` | ✅ |
| PlannerAgent | `planner_agent.py`, `prompts/planner.txt` | ✅ |
| SchemaRetrieverTool | `sql_retrieval_agent.py`, `prompts/sql_retrieval.txt` | ✅ |
| SQLGeneratorAgent | `sqlgenerator_agent.py`, `prompts/sglgenerator.txt` | ✅ |
| VerifierAgent | `verifier_agent.py` | ✅ |
| MEP schema | `mep/schema.py` | ✅ |
| MEP writer / builder | `mep/writer.py` | ⚠️ Stubs not implemented |
| Pipeline runner | `runner/run_generate_meps.py` | ✅ |
| eval_outputs (Pass 1) | `eval/eval_outputs.py` | ✅ |
| eval_traces (Pass 2) | `eval/eval_traces.py` | ✅ |
| eval_topk (Pass 3) | `eval/eval_topk.py` | ⚠️ Needs rewrite |
| LLM judge | `eval/judge.py` | ⚠️ Rubric needs update |
| error_taxonomy (Pass 4) | `eval/error_taxonomy.py` | ⚠️ Needs SQL taxonomy |
| summarize.py | `eval/summarize.py` | ✅ |
| HTML report | `eval/report.py` | ⚠️ Needs update |
| Dashboard | `eval/dashboard.py` | ⚠️ Needs update |
| DB setup | `eval/db_setup.py` | ✅ |
| Langfuse tracing | `langfuse_integration/tracing.py`, `client.py` | ✅ |
| Langfuse prompts | `langfuse_integration/prompts.py` | ⚠️ Still registers vision.txt |
| Langfuse dataset | `langfuse_integration/dataset.py` | ⚠️ May point to ChartQAPro |
| Shared utils | `utils/json_strict.py`, `utils/timing.py` | ✅ |
| **Legacy (unused)** | `vision_agent.py`, `vision_qa_tool.py`, `ocr_reader_tool.py`, `prompts/vision.txt`, `chartqapro_loader.py`, `utils/hashing.py` | 🟡 Not used — clean up |
