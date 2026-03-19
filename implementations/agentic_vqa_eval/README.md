# Agentic ChartQAPro Evaluation Framework

## Introduction

Welcome to **Reference Implementation 6** of the Survey Paper on Agentic Visual Question Answering. This implementation explores how multi-agent systems built on **CrewAI** can evaluate and interpret chart-based questions from the **ChartQAPro** dataset. Unlike single-pass VLM approaches, this framework decomposes chart QA into an explicit **Plan → Inspect → Explain** loop, producing fully traceable evaluation artifacts for each sample.

The core contribution is the **Model Evaluation Packet (MEP)** — a portable JSON trace that captures everything: the inspection plan, the vision agent's reasoning, the verifier's critique, tool call logs, timestamps, and errors. This enables reproducible evaluation, post-hoc explainability analysis, and model comparison across VLM backends.

**Observability layer:** Integration with **[Langfuse](https://langfuse.com)** (cloud or self-hosted) for live trace visualization, prompt versioning, dataset registration, and experiment comparison across configs — all without changing the MEP ground-truth artifacts.

---

## Architecture Overview

<details>
<summary>Show pipeline diagram</summary>

```
┌──────────────────────────────────────────────────────────┐
│                     Input Sample                          │
│         (question, chart image, expected answer)          │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                   PlannerAgent  (text-only LLM)           │
│                                                           │
│  • Receives: question + question type                     │
│  • Produces: structured JSON inspection plan              │
│    {                                                       │
│      "steps": ["Identify chart type...",                  │
│                "Locate axes and legend...",               │
│                "Extract data values...",                  │
│                "Compare / compute answer..."],            │
│      "expected_answer_type": "number",                    │
│      "answerability_check": "answerable"                  │
│    }                                                       │
│  • Does NOT see the image — pure symbolic planning        │
└─────────────────────────┬────────────────────────────────┘
                          │  plan.steps
                          ▼
┌──────────────────────────────────────────────────────────┐
│           OcrReaderTool  (optional — enabled by default)  │
│                                                           │
│  • Receives: chart image path                             │
│  • Makes a single VLM call focused purely on text         │
│    transcription — no reasoning, no question              │
│  • Produces: structured JSON of all visible text          │
│    {                                                       │
│      "chart_type": "bar",                                 │
│      "title": "...",                                      │
│      "x_axis": {"label": "...", "ticks": [...]},          │
│      "y_axis": {"label": "...", "ticks": [...]},          │
│      "legend": [...], "data_labels": [...],               │
│      "annotations": [...]                                 │
│    }                                                       │
│  • Separates perception from reasoning — VisionAgent gets │
│    pre-extracted text as grounding context, reducing      │
│    hallucinated axis labels and misread tick values       │
│  • Can be skipped with --no_ocr                           │
└─────────────────────────┬────────────────────────────────┘
                          │  ocr grounding context
                          ▼
┌──────────────────────────────────────────────────────────┐
│                   VisionAgent  (multimodal LLM + tool)    │
│                                                           │
│  • Receives: image path, question, plan steps             │
│  • Calls vision_qa_tool ONCE (enforced)                   │
│    └─► VisionQATool:                                      │
│         - encodes image as base64                         │
│         - calls OpenAI / Gemini vision API                │
│         - logs full tool trace (tokens, latency, req_id)  │
│  • Produces: draft JSON answer + grounded explanation     │
│    {"answer": "3", "explanation": "The chart shows..."}   │
└─────────────────────────┬────────────────────────────────┘
                          │  draft answer + explanation
                          ▼
┌──────────────────────────────────────────────────────────┐
│              VerifierAgent  (Pass 2.5 — VLM critique)     │
│                                                           │
│  • Receives: chart image, question, plan steps,           │
│              draft answer, draft explanation              │
│  • Makes a single direct VLM call (no tool use)          │
│  • Independently re-examines the chart image             │
│  • Decides: CONFIRM or REVISE the draft answer           │
│  • Produces: {"verdict": "confirmed" | "revised",        │
│               "answer": "<final>",                        │
│               "reasoning": "<one visual sentence>"}       │
│  • Can be skipped with --no_verifier                      │
└─────────────────────────┬────────────────────────────────┘
                          │  final answer
                          ▼
┌──────────────────────────────────────────────────────────┐
│               Model Evaluation Packet (MEP)               │
│                                                           │
│  Saved as:  meps/<config>/<dataset>/<split>/<id>.json    │
│  Contains:  schema_version, run_id, config, sample,       │
│             plan, ocr, vision, verifier, timestamps, errors│
└─────────────────────────┬────────────────────────────────┘
                          │
           ┌──────────────┼──────────────────┐
           ▼              ▼                  ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │ eval_outputs │ │ eval_traces  │ │  eval_topk   │
   │              │ │              │ │              │
   │ • accuracy   │ │ • latency    │ │ • hit@1/2/3  │
   │ • LLM judge  │ │ • tool calls │ │ (top-K pass) │
   │   rubric     │ │ • replayabil │ │              │
   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
          └────────────────┴────────────────┘
                           │
                           ▼
                    summarize.py
               (summary.csv by config × question_type)
```

</details>

### Explainability at Four Levels

This framework produces explainability signals at four distinct levels:

1. **Process-level** — The MEP trace captures every step: what the planner decided, which tool was called, what inputs it received, and what the VLM returned. Any run can be fully replayed.

2. **Output-level** — The VisionAgent is required to produce a natural language `explanation` grounded in specific chart evidence (axis labels, legend entries, data values, trend direction). Hallucinated or vague explanations are not compliant.

3. **Quality-level** — The LLM-as-judge (`eval/judge.py`) scores five rubric dimensions:
   - `explanation_quality` — is the explanation specific and chart-grounded?
   - `hallucination_rate` — does it claim things not visible in the chart?
   - `plan_coverage` — does the explanation address each inspection step?
   - `plan_adherence` — were steps followed in order?
   - `faithfulness_alignment` — does the explanation logically support the answer?

4. **Critique-level** — The VerifierAgent (Pass 2.5) independently re-examines the chart image and the VisionAgent's draft, producing a one-sentence visual rationale for why it confirmed or revised the answer. This is stored in `verifier.parsed.reasoning` in the MEP and is directly readable alongside the failure taxonomy.

---

## Prerequisites

- Python **3.10** or higher
- A virtual environment manager (`venv`)
- An **OpenAI API key** (for `openai_openai` and `openai_gemini` configs)
- A **Google Gemini API key** (for `gemini_gemini` and `gemini_openai` configs)
- Sufficient disk space for chart images (~50 MB for full ChartQAPro test split)

---

## Package Dependencies

<details>
<summary>Show dependencies table</summary>

| Package | Version | Purpose |
|---|---|---|
| `crewai` | 1.10.1 | Multi-agent framework: Agent, Task, Crew, LLM, BaseTool |
| `openai` | 2.26.0 | GPT-4o planner and vision inference |
| `google-generativeai` | 0.8.6 | Gemini planner and vision inference |
| `datasets` | 4.6.1 | HuggingFace dataset loader for ChartQAPro |
| `pillow` | 12.1.1 | Image handling and format conversion |
| `pydantic` | 2.11.10 | Tool input validation and schema enforcement |
| `json_repair` | 0.25.3 | Fallback JSON parsing when LLM output is malformed |
| `python-dotenv` | 1.1.1 | API key management via `.env` file |
| `pandas` | 2.3.3 | Metric aggregation and summary CSV generation |
| `langfuse` | latest | Trace visualization, prompt versioning, dataset registration |
| `matplotlib` | ≥3.7 | Charts in notebook and dashboard |
| `streamlit` | ≥1.32 | Interactive evaluation dashboard |
| `jupyter` / `ipykernel` | latest | Analysis notebook |

</details>

---

## Internal Package Structure

<details>
<summary>Show package tree</summary>

```
src/agentic_chartqapro_eval/
├── utils/
│   ├── json_strict.py      — Strict JSON parser with json_repair fallback
│   ├── hashing.py          — SHA-256 image integrity tracking
│   └── timing.py           — timed() context manager, ISO timestamp helper
│
├── datasets/
│   ├── perceived_sample.py — PerceivedSample dataclass, QuestionType enum
│   └── chartqapro_loader.py — HuggingFace loader, image saving, type mapping
│
├── mep/
│   ├── schema.py           — MEP, MEPConfig, MEPSample, MEPPlan, MEPVision, MEPVerifier dataclasses
│   └── writer.py           — write_mep(), iter_meps() for I/O
│
├── agents/
│   ├── planner_agent.py    — PlannerAgent: text-only LLM, returns JSON plan
│   ├── vision_agent.py     — VisionAgent: multimodal LLM + tool, returns draft answer + explanation
│   ├── verifier_agent.py   — VerifierAgent (Pass 2.5): direct VLM critique, confirms or revises draft
│   └── prompts/
│       ├── planner.txt     — Planner system prompt template
│       └── vision.txt      — Vision agent system prompt template
│
├── tools/
│   ├── vision_qa_tool.py   — VisionQATool (CrewAI BaseTool), OpenAI + Gemini backends
│   └── ocr_reader_tool.py  — OcrReaderTool (CrewAI BaseTool): text-only transcription, optional pre-read step
│
├── runner/
│   └── run_generate_meps.py — Main pipeline: loads dataset, runs agents, writes MEPs
│
├── eval/
│   ├── judge.py            — LLM-as-judge: 5-rubric scoring via text LLM
│   ├── eval_outputs.py     — Pass 1: rule-based accuracy + judge rubric
│   ├── eval_traces.py      — Pass 2: latency, tool call count, replayability
│   ├── eval_topk.py        — Top-K pass: hit@1/2/3 by re-querying VLM for candidates
│   ├── error_taxonomy.py   — Pass 4: VLM-based failure classification (axis_misread, arithmetic_mistake, …)
│   ├── report.py           — HTML report generator: summary cards, charts, per-sample table
│   ├── dashboard.py        — Streamlit interactive dashboard: sample browser, chart image viewer
│   └── summarize.py        — Aggregate metrics.jsonl → summary.csv
│
└── langfuse_integration/
    ├── client.py           — Langfuse client singleton (gracefully disabled if not configured)
    ├── tracing.py          — sample_trace(), open_llm_span(), close_span() helpers
    ├── prompts.py          — Push planner.txt / vision.txt to Langfuse Prompt Management
    ├── dataset.py          — Register ChartQAPro samples as a Langfuse Dataset
    └── ingest.py           — Retroactively import existing MEP files into Langfuse
```

</details>

---

## Getting Started

### 1. Install dependencies

From the **root of the repository**, install the `agentic-xai-eval` dependency group using `uv`:

```bash
uv sync --group agentic-xai-eval
source .venv/bin/activate
```

The `agentic_chartqapro_eval` package is automatically available — it is included in the root package's build configuration and installed as part of the sync.

### 2. Configure API keys

```bash
# From the repo root:
cp .env.example .env
# Edit .env and fill in your keys:
#   OPENAI_API_KEY=...
#   GEMINI_API_KEY=...
#   LANGFUSE_PUBLIC_KEY=...   # optional — for observability
#   LANGFUSE_SECRET_KEY=...
```

The `.env` file lives at the **repo root**. `load_dotenv()` searches upward from the working directory, so it is found automatically regardless of which subdirectory you run commands from.

### 3. Run integration tests to validate your API keys

```bash
uv run --env-file .env pytest -sv tests/tool_tests/test_integration.py
```

> **Note:** If your `.env` file is incomplete or needs to be updated, you can re-run onboarding manually from inside your Coder workspace (from the repo root):
>
> ```bash
> onboard --bootcamp-name "llm-interpretability-bootcamp" --output-dir "." --test-script "./aieng-llm-interp/tests/test_integration.py" --env-example "./.env.example" --test-marker "integration_test" --force
> ```

### 4. Generate MEPs (run the agentic pipeline)

> **Note:** All `uv run` commands below use `$(git rev-parse --show-toplevel)` so they work from any directory in the repo — it resolves the repo root for `--env-file`, while `--directory` ensures outputs (`meps/`, `output/`) are written inside `implementations/agentic_vqa_eval/`.

Run on 25 test samples using GPT-4o for planner, vision, and verifier:

```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.runner.run_generate_meps \
    --split test \
    --n 25 \
    --config gemini_gemini \
    --workers 4 \
    --out meps/
```

MEPs are written to `meps/gemini_gemini/chartqapro/test/<sample_id>.json`.

The **VerifierAgent (Pass 2.5)** runs automatically after the VisionAgent on every sample. To skip it (faster, lower cost):
```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.runner.run_generate_meps \
    --split test --n 25 --config gemini_gemini --no_verifier
```

**Available configs:** `openai_openai`, `gemini_gemini`, `openai_gemini`, `gemini_openai`

**Model overrides** (e.g. to test different models without changing config):
```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.runner.run_generate_meps \
    --split test --n 25 --config gemini_gemini \
    --planner_model gemini-2.5-flash-lite \
    --vision_model gemini-2.5-flash-lite \
    --verifier_model gemini-2.5-flash-lite   # defaults to vision_model if omitted
```

#### OCR pre-reader (optional)

The **OcrReaderTool** runs between the Planner and VisionAgent. It makes a single VLM call to transcribe all visible text from the chart image (axis labels, tick values, legend entries, title, data labels, annotations) into structured JSON. This JSON is injected into the VisionAgent's prompt as grounding context, separating *perception* (what text is in the chart) from *reasoning* (what the answer is). This reduces hallucinated axis values and misread tick labels.

OCR is **enabled by default** and uses the same vision backend and model as the VisionAgent.

To run with OCR using a cheaper model (recommended — OCR is simpler than full VQA):
```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.runner.run_generate_meps \
    --split test --n 25 --config gemini_gemini \
    --ocr_model gemini-2.5-flash-lite
```

To disable OCR entirely (matches the original pipeline behaviour, faster and lower cost):
```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.runner.run_generate_meps \
    --split test --n 25 --config gemini_gemini --no_ocr
```

When OCR is skipped, `"ocr": null` appears in the MEP and `"ocr_ms": 0.0` in timestamps — consistent with how `--no_verifier` behaves.

**Context injection:** The VisionAgent uses a single shared prompt template (`agents/prompts/vision.txt`) that contains an `{ocr_block}` placeholder. When OCR ran successfully, this block is populated with the structured OCR fields (chart type, title, axis labels, legend). When OCR is skipped or produced no output, `{ocr_block}` renders as an empty string — the prompt is otherwise identical. This is a useful example of conditional context injection: the same template handles both modes without branching at the prompt level.

### 5. Evaluate outputs (Pass 1 — accuracy + judge)

```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.eval.eval_outputs \
    --mep_dir meps/gemini_gemini/chartqapro/test \
    --out output/metrics.jsonl \
    --no_judge          # omit this flag to enable LLM judge (costs API calls)
```

Each line in `metrics.jsonl` contains: `sample_id`, `expected`, `predicted`, `answer_accuracy`, `latency_sec`, `tool_call_count`, plus `judge_*` scores if enabled.

When the verifier ran, two extra columns are present:
- `vision_answer` — the VisionAgent's raw draft answer before verification
- `verifier_verdict` — `"confirmed"` (verifier agreed) or `"revised"` (verifier corrected), or `"skipped"` if `--no_verifier` was used

The `predicted` column always reflects the **final answer** — the verifier's output when it ran, or the vision agent's output when skipped. This means accuracy scores automatically capture any corrections made by the verifier.

### 6. Evaluate traces (Pass 2 — latency and replayability)

```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.eval.eval_traces \
    --mep_dir meps/gemini_gemini/chartqapro/test \
    --out output/trace_metrics.jsonl
```

### 7. Run Top-K evaluation (hit@1/2/3)

Re-queries the VLM for each MEP asking for the 3 most likely candidate answers:

```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.eval.eval_topk \
    --mep_dir meps/gemini_gemini/chartqapro/test \
    --out output/topk_metrics.jsonl \
    --backend gemini \
    --model gemini-2.5-flash-lite \
    --k 3
```

This pass does **not** modify existing MEPs or `metrics.jsonl`.

### 8. Summarize results

```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.eval.summarize \
    --metrics output/metrics.jsonl \
    --out output/summary.csv
```

### 9. Failure taxonomy (Pass 4 — VLM-based diagnosis)

This pass asks **why** the agent was wrong, not just **that** it was wrong. A VLM is given the original chart image alongside the wrong answer, the correct answer, the agent's explanation, and the inspection plan — so it can make a *visual* diagnosis of the failure mode.

```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.eval.error_taxonomy \
    --mep_dir meps/gemini_gemini/chartqapro/test \
    --metrics_file output/metrics.jsonl \
    --out output/taxonomy.jsonl
```

Each line in `taxonomy.jsonl` contains a `failure_type` (one of the categories below) and a `failure_reason` sentence grounded in what the VLM observed in the chart.

**Failure categories:**

<details>
<summary>Show failure categories</summary>

| Category | Description |
|---|---|
| `correct` | Model got it right — no VLM call made |
| `axis_misread` | Read the wrong axis value, scale, or unit |
| `legend_confusion` | Mixed up series, colours, or legend entries |
| `arithmetic_mistake` | Extracted correct data but made a calculation error |
| `hallucinated_element` | Referenced data or labels not visible in the chart |
| `unanswerable_failure` | Should have said UNANSWERABLE but didn't (or vice versa) |
| `question_misunderstanding` | Answered a different or adjacent question |
| `extraction_error` | Could not locate the relevant data in the chart at all |
| `other` | Does not fit any category above |

</details>

**Why VLM instead of text-only LLM?**
A text-only judge can only read the agent's description of what it saw. A VLM can independently verify whether the axis labels were actually ambiguous, whether the cited data point actually appears in the image, or whether the legend entries are genuinely confusing — producing a grounded diagnosis rather than a guess.

**Quick breakdown after running:**
```bash
python -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('output/taxonomy.jsonl')]
for k, n in Counter(r['failure_type'] for r in rows).most_common():
    print(f'{k:<30} {n}')
"
```

**Cross-referencing with the verifier (Pass 2.5):** After running both passes, you can ask which failure types the verifier caught vs. missed. Filter `metrics.jsonl` for rows where `verifier_verdict == "revised"` and join on `sample_id` with `taxonomy.jsonl` to see the distribution:

```bash
python -c "
import json

metrics = {r['sample_id']: r for line in open('output/metrics.jsonl') for r in [json.loads(line)]}
taxonomy = {r['sample_id']: r for line in open('output/taxonomy.jsonl') for r in [json.loads(line)]}

revised = [sid for sid, m in metrics.items() if m.get('verifier_verdict') == 'revised']
print(f'Verifier revised {len(revised)} answers')
for sid in revised:
    t = taxonomy.get(sid, {})
    print(f'  {sid}: {t.get(\"failure_type\", \"?\")} — {t.get(\"failure_reason\", \"\")}')
"
```

### 10. Visualization & Reporting

#### HTML report (no extra dependencies)

Generates a single portable HTML file with summary cards, accuracy tables, verifier stats, failure taxonomy breakdown, and a per-sample results table:

```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.eval.report \
    --metrics output/metrics.jsonl \
    --taxonomy output/taxonomy.jsonl \
    --out output/report.html
open output/report.html   # macOS; use xdg-open on Linux
```

The report is fully self-contained — one file you can email or commit.

#### Streamlit dashboard (interactive)

Launch:
```bash
streamlit run src/agentic_chartqapro_eval/eval/dashboard.py
```

The dashboard opens at **http://localhost:8501** and provides:
- **Overview tab** — summary metrics, accuracy-by-type bar chart, verifier verdict distribution, failure taxonomy, judge score histograms. All panels respond to the sidebar filters (question type, verdict, failure type).
- **Sample Browser tab** — select any sample by ID, see the chart image inline, read the plan steps, vision draft, verifier verdict, taxonomy diagnosis, and per-span latency.

#### Jupyter notebooks

Two notebooks are provided, covering opposite ends of the workflow:

| Notebook | Purpose |
|---|---|
| `run_pipeline.ipynb` | **Execution** — generates MEPs and runs all eval passes end-to-end from a single config cell |
| `analysis.ipynb` | **Analysis** — visualises existing results (accuracy charts, verifier stats, failure taxonomy, per-sample browser) |

**Execution notebook** (start here):
```bash
# ./.venv/bin/jupyter notebook
run_pipeline.ipynb
```
Edit the configuration cell at the top (N_SAMPLES, CONFIG, USE_OCR, USE_VERIFIER), then *Run All*. Covers: dataset loading → agent instantiation → pipeline run → OCR ablation → Pass 1/2/4 evaluation → summary CSV.

**Analysis notebook** (after generating results):
```bash
# ./.venv/bin/jupyter notebook
analysis.ipynb
```
Pre-built cells walk through: loading MEPs, accuracy by question type, verifier before/after comparison, failure taxonomy chart, judge score distributions, and a single-sample deep-dive with inline chart image.

---

## Langfuse Observability

Langfuse is an open-source LLM observability platform that adds a live visualization and experiment-comparison layer on top of the MEP artifacts. MEPs remain the portable ground truth; Langfuse is purely additive.

> **Optional:** If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are not set in `.env`, all Langfuse calls are silent no-ops and the pipeline runs exactly as before.

<details>
<summary>Show Langfuse setup and usage</summary>

### What Langfuse gives you

| Feature | Detail |
|---|---|
| **Trace viewer** | Every sample becomes a trace with `planner` and `vision_qa_tool` child generations showing prompts, outputs, token usage, and latency |
| **Feedback scores** | `answer_accuracy` and all five `judge_*` rubric scores are attached to each trace after eval |
| **Prompt Management** | `planner.txt` and `vision.txt` are versioned — every experiment links to the exact prompt version used |
| **Dataset registry** | ChartQAPro samples are registered so experiments formally reference a dataset version |
| **Experiment comparison** | `openai_openai` vs `gemini_gemini` side-by-side with accuracy distributions and latency CDFs |

### Trace structure

```
Trace: chartqapro/000002  [openai_openai | standard | 11.4s]
  input:    {question, expected_output}
  output:   {answer, explanation}
  scores:   answer_accuracy=1.0, judge_explanation_quality=0.9, ...
  ├── Generation: planner          [gpt-4o | 2.1s]
  │     input: {prompt}
  │     output: {plan_steps: [...], parse_error: false}
  ├── Generation: vision_agent     [gpt-4o | 5.6s]
  │     └── Generation: vision_qa_tool  [gpt-4o | 2.9s | 688 tokens]
  │           input:  {image_path, question, plan_steps}
  │           output: {answer, explanation}
  └── Generation: verifier         [gpt-4o | 3.7s]
        input:  {prompt, draft_answer}
        output: {verdict: "confirmed" | "revised", answer, reasoning}
```

### 1. Get API keys

**Cloud (recommended — no infrastructure needed):**

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a new project
3. Go to **Settings → API Keys** and create a key pair

**Self-hosted:**

Follow the [Langfuse self-hosting guide](https://langfuse.com/docs/deployment/self-host) to deploy with Docker Compose, then create API keys in the UI.

### 2. Configure the connection

Add to your `.env` at the repo root:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
# LANGFUSE_HOST=https://cloud.langfuse.com  # default; change for self-hosted
```

### 3. Push prompt versions to Langfuse

Run once before starting experiments. This creates versioned entries for `planner.txt` and `vision.txt` in Langfuse Prompt Management so every future experiment links to the exact prompt version used.

```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.langfuse_integration.prompts
```

### 4. Register the dataset

```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.langfuse_integration.dataset \
    --split test --n 25
```

This creates a dataset named `ChartQAPro_test` in Langfuse containing one item per sample (question, expected output, question type, image path).

### 5. Live tracing (automatic on new runs)

No extra flags needed. When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, the pipeline automatically registers the dataset and versions the prompts at run start, opens a Langfuse trace per sample, creates child generations with inputs/outputs/token usage, and stores the `lf_trace_id` in the MEP for later score attachment.

```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.runner.run_generate_meps \
    --split test --n 25 --config gemini_gemini --workers 4 --out meps/
```

### 6. Attach evaluation scores

After running `eval_outputs.py`, accuracy and judge scores are automatically written back to the Langfuse traces:

```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.eval.eval_outputs \
    --mep_dir meps/gemini_gemini/chartqapro/test \
    --out metrics.jsonl
```

### 7. Ingest existing MEPs (retroactive)

If you have MEPs from runs before Langfuse was configured, import them without re-running the pipeline:

```bash
uv run --env-file "$(git rev-parse --show-toplevel)/.env" --directory "$(git rev-parse --show-toplevel)/implementations/agentic_vqa_eval" -m agentic_chartqapro_eval.langfuse_integration.ingest \
    --mep_dir meps/gemini_gemini/chartqapro/test \
    --metrics_file metrics.jsonl   # optional: attaches scores if available
```

</details>

---

## MEP Schema

<details>
<summary>Show full MEP schema and field reference</summary>

Each MEP file is a self-contained JSON evaluation artifact:

```json
{
  "schema_version": "mep.v1",
  "run_id": "<uuid>",
  "config": {
    "planner_backend": "openai",
    "vision_backend": "openai",
    "judge_backend": "openai",
    "config_name": "openai_openai",
    "planner_model": "gpt-4o",
    "vision_model": "gpt-4o"
  },
  "sample": {
    "sample_id": "chartqapro_000002",
    "question": "how many times did retail sales growth fall below average by more than 2%",
    "question_type": "standard",
    "expected_output": "3",
    "image_ref": { "path": "...", "sha256": "..." }
  },
  "plan": {
    "parsed": { "steps": [...], "answerability_check": "answerable" },
    "parse_error": false
  },
  "ocr": {
    "parsed": {
      "chart_type": "bar",
      "title": "Retail Sales Growth vs Average",
      "x_axis": {"label": "Year", "ticks": ["2018", "2019", "2020", "2021", "2022"]},
      "y_axis": {"label": "Growth (%)", "ticks": ["-4", "-2", "0", "2", "4", "6"]},
      "legend": ["Retail Sales Growth", "Average"],
      "data_labels": [],
      "annotations": []
    },
    "parse_error": false,
    "tool_trace": [{ "tool": "ocr_reader_tool", "elapsed_ms": 1243, "usage": {...} }]
  },
  "vision": {
    "parsed": { "answer": "4", "explanation": "The chart shows..." },
    "tool_trace": [{ "tool": "vision_qa_tool", "elapsed_ms": 2948, "usage": {...} }],
    "parse_error": false
  },
  "verifier": {
    "parsed": {
      "verdict": "revised",
      "answer": "3",
      "reasoning": "The chart clearly shows three bars below the dashed average line, not four."
    },
    "verdict": "revised",
    "parse_error": false
  },
  "timestamps": { "planner_ms": 2185, "ocr_ms": 1243, "vision_ms": 5684, "verifier_ms": 3712 },
  "errors": [],
  "lf_trace_id": "abc123..."
}
```

`ocr` is `null` when `--no_ocr` is passed. When present, `ocr.parsed` contains: `chart_type`, `title`, `x_axis`, `y_axis`, `legend`, `data_labels`, `annotations`.

`verifier` is `null` when `--no_verifier` was passed. When present, `verifier.verdict` is one of:

- `"confirmed"` — second model agreed with the draft answer
- `"revised"` — second model caught an error and corrected the answer
- `"skipped"` — verifier ran but fell back due to missing image or error

</details>

---

## Resources

- **ChartQAPro Dataset** — Chart question answering benchmark with factoid, MCQ, conversational, hypothetical, and unanswerable question types ([HuggingFace](https://huggingface.co/datasets/ahmed-masry/ChartQAPro))
- **CrewAI Documentation** — Multi-agent orchestration framework used for PlannerAgent and VisionAgent ([docs.crewai.com](https://docs.crewai.com))
- **OpenAI Vision API** — GPT-4o multimodal inference for chart image understanding ([platform.openai.com](https://platform.openai.com/docs))
- **Google Gemini API** — Alternative VLM backend for vision inference ([ai.google.dev](https://ai.google.dev/docs))
- **LLM-as-Judge (Zheng et al., 2023)** — Methodology for using LLMs to score free-form outputs with rubric dimensions ([arXiv:2306.05685](https://arxiv.org/abs/2306.05685))
- **Langfuse** — Open-source LLM observability platform used for tracing, prompt versioning, and experiment comparison ([langfuse.com](https://langfuse.com))

---

## FAQ

<details>
<summary>Show all FAQs</summary>

### 1. What is the purpose of the MEP schema?

The Model Evaluation Packet (MEP) schema is designed to provide a comprehensive, portable, and reproducible trace of the evaluation process. It captures all relevant details, including the inspection plan, tool calls, timestamps, and errors, enabling post-hoc analysis and comparison across models.

### 2. Can I use a different dataset with this framework?

Yes, the framework is modular and supports other datasets as long as they are compatible with the expected input format (question, chart image, expected answer). You may need to implement a custom dataset loader in `src/agentic_chartqapro_eval/datasets/`.

### 3. How do I add a new vision or planner backend?

To add a new backend, you need to:

- Implement the corresponding tool or agent in `src/agentic_chartqapro_eval/tools/` or `src/agentic_chartqapro_eval/agents/`.
- Update the configuration options in `run_generate_meps.py` to include the new backend.

### 4. What happens if the VisionAgent produces malformed JSON?

The framework uses the `json_repair` library to attempt to fix malformed JSON outputs. If repair fails, the error is logged in the MEP under the `errors` field.

### 5. How can I customize the evaluation rubric?

The evaluation rubric is defined in `src/agentic_chartqapro_eval/eval/judge.py`. You can modify the scoring dimensions or add new ones by editing the `judge` function.

### 6. Is it possible to run the framework without API calls?

Yes, you can use pre-generated MEPs for evaluation by skipping the generation step. This is useful for offline analysis or when API usage is restricted.

### 7. How do I handle large datasets efficiently?

For large datasets, consider:

- Using the `--n` flag to process a subset of samples.
- Increasing the `--workers` count to parallelize processing.
- Running the pipeline on a machine with sufficient memory and disk space.

### 8. Where can I find more examples or tutorials?

Refer to the Resources section for links to documentation, datasets, and related research papers. Additional examples may be added in future updates.

### 9. How does the VerifierAgent differ from the LLM judge?

They serve different purposes and run at different times:

| | VerifierAgent (Pass 2.5) | LLM Judge (Pass 1) |
|---|---|---|
| **When** | During MEP generation — before the MEP is written | After MEP generation — reads an existing MEP |
| **Can see chart image** | Yes — direct VLM call with base64 image | No — text-only |
| **Output** | Corrects the answer in real-time (`verdict: "revised"`) | Scores the quality of the explanation |
| **Effect on accuracy** | Changes `predicted` if it revises | Only adds `judge_*` score columns |
| **Primary purpose** | Catch visual errors before they are recorded | Evaluate how well the explanation is grounded |

The verifier improves the pipeline's answer quality; the judge measures the pipeline's reasoning quality.

### 10. Do I need Langfuse to run the framework?

No. Langfuse is entirely optional. If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are not set in `.env`, all Langfuse calls are silent no-ops. The pipeline produces the same MEPs, `metrics.jsonl`, and `summary.csv` as before.

</details>
