# Interpretability for LLMs and Agents Bootcamp

This repository contains reference implementations created by the Vector AI Engineering team
for the **Interpretability for LLMs and Agents Bootcamp** — a hands-on program exploring interpretability,
fairness, alignment, and agentic evaluation of large language and vision-language models.

## About This Bootcamp

The bootcamp covers six core topics spanning the modern AI interpretability and evaluation
landscape. Each implementation is a self-contained reference that demonstrates techniques from
recent research, with fully reproducible notebooks and evaluation pipelines.

## Repository Structure

- **docs/**: Additional documentation and setup guides.
- **implementations/**: One directory per topic, each containing notebooks and a README.
- **pyproject.toml**: Centralizes project settings, build requirements, and dependencies.
- **scripts/**: Utility scripts for environment setup and data preparation.

### Implementations

| # | Topic | Description |
|---|-------|-------------|
| 1 | [XAI Refresher](implementations/xai_refresher/) | Foundations of explainable AI — feature attribution, saliency maps, and model-agnostic explanation methods |
| 2 | [Bias & Fairness Analysis](implementations/bias_fairness_analysis/) | Detecting and mitigating bias in ML models across demographic groups |
| 3 | [Preference Alignment](implementations/preference_alignment/) | LLM alignment with human preferences using DPO framework |
| 4 | [Multimedia RAG + VLM](implementations/multimedia_rag/) | Cross-modal retrieval-augmented generation with ImageBind (audio, video, text) |
| 5 | [Agentic ChartQA Evaluation](implementations/agentic_vqa_eval/) | Multi-agent evaluation harness for chart-based VQA using CrewAI and ChartQAPro |
| 6 | [Mechanistic Interpretability](implementations/mechanistic_interpretability/) | Sparse Autoencoders for LLM feature discovery, and logit-lens + activation patching for VLM modality fusion |

## Getting Started

1. Clone this repository:

   ```bash
   git clone <repo-url>
   cd interpretability_agent_bootcamp
   ```

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Install dependencies for the topic you want to work with. All dependency groups are defined
   in the root `pyproject.toml` — install only the group(s) you need:

   | Topic | Group name | Install command |
   | ------- | ----------- | ----------------- |
   | XAI Refresher | `xai-refresher` | `uv sync --group xai-refresher` |
   | Bias & Fairness Analysis | None | `uv sync` |
   | Preference Alignment (DPO) | `preference-alignment` | `uv sync --group preference-alignment` |
   | Multimedia RAG | `multimedia-rag` | `uv sync --group multimedia-rag` |
   | Agentic ChartQA Eval | `agentic-xai-eval` | `uv sync --group agentic-xai-eval` |
   | Mechanistic Interpretability | `mechanistic-interp` | `uv sync --group mechanistic-interp` |

   > **Conflict note:** The `mechanistic-interp` and `xai-refresher` groups cannot be installed
   > together — they have conflicting `datasets` package requirements. Install only one at a time.

   > **CUDA note (ref4 — Preference Alignment):** The group uses `torch==2.6.0` from PyPI
   > (which includes CUDA support on Linux). If you specifically need the CUDA 12.4 build, run:

   > ```bash
   > uv sync --group preference-alignment \
   >   --index-url https://download.pytorch.org/whl/cu124
   > ```

4. Launch JupyterLab and open the notebooks in the relevant `implementations/<topic>/` directory:

   ```bash
   uv run jupyter lab
   ```

5. Run integration tests to validate that your API keys are set up correctly:

   ```bash
   uv run --env-file .env pytest -sv tests/test_integration.py
   ```

   > **Note:** If your `.env` file is incomplete or needs to be updated, you can re-run onboarding manually from inside your Coder workspace (from the repo root):
   >
   > ```bash
   > onboard --bootcamp-name "llm-interpretability-bootcamp" --output-dir "." --test-script "./aieng-llm-interp/tests/test_integration.py" --env-example "./.env.example" --test-marker "integration_test" --force
   > ```

## License

This project is licensed under the terms of the [LICENSE](LICENSE.md) file in the root directory.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting pull requests.

## Contact

For questions or help navigating this repository, contact Aravind Narayanan at
[aravind.narayanan@vectorinstitute.ai](mailto:aravind.narayanan@vectorinstitute.ai)
