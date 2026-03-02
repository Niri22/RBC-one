# Bias & Fairness Analysis Overview

## Introduction

Welcome to the **Bias & Fairness Analysis** implementation of the Interpretability Agent Bootcamp.

This reference implementation centers on a **notebook-driven workflow** for:

* Zero-shot classification using large language models (LLMs)
* Token-level interpretability via Integrated Gradients (IG)
* Group-based fairness analysis across demographic identities

The goal is to **analyze and interpret model bias patterns**, not to evaluate a specific content category. The emphasis is on understanding model behavior and surfacing disparities across identity groups.

---

## Environment Setup (Recommended)

This project uses a Python 3.10 environment.

### Option A — Using `uv` (recommended)

```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync --active
python -m ipykernel install --user \
  --name bias-fairness-analysis \
  --display-name "Bias & Fairness (Py3.10)"
```

### Option B — Using `venv` + pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m ipykernel install --user \
  --name bias-fairness-analysis \
  --display-name "Bias & Fairness (Py3.10)"
```

---

## Dataset Setup

Datasets are **not included in this repository**. They are downloaded or prepared dynamically within the notebook.

### Supported Datasets

### 1. CivilComments

* Downloaded automatically via:

  ```python
  load_dataset("google/civil_comments")
  ```

* No manual download required.

* Processed file is saved to:

  ```
  data/civil/civil.parquet
  ```

---

### 2. Jigsaw (Unintended Bias Dataset)

* Must be downloaded manually from Kaggle (https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data?select=train.csv)

* After downloading, place:

  ```
  train.csv
  ```

  inside:

  ```
  data/jigsaw/
  ```

* The notebook converts it to Parquet format automatically.

---

## Notebook-First Workflow

This implementation is designed to be used primarily through:

```
zero_shot_hate_explanations.ipynb
```

### Launch Jupyter

```bash
jupyter lab
```

Open:

```
zero_shot_hate_explanations.ipynb
```

---

## Notebook Configuration

Inside the notebook, configure:

```python
DATASET = "civil"   # or "jigsaw"
STREAM = True
TAKE = 50000
SAMPLE = None

DATA_DIR = f"./data/{DATASET}/"
OUT_PATH = f"./data/{DATASET}/{DATASET}.parquet"
```

The notebook will:

* Create required directories
* Download or process the dataset
* Save processed data locally
* Run zero-shot classification
* Generate Integrated Gradients explanations
* Compute fairness metrics (SPD, EOpp)
* Produce group-level and worst-case summaries

---

## Core Components (Supporting Modules)

Although the notebook is the primary interface, the following modules provide reusable functionality:

* **download_data.py**
  Dataset normalization and Parquet conversion utilities.

* **llm_zero_shot_explain.py**
  Zero-shot classification + Integrated Gradients implementation.

* **fairness_metrics.py**
  Fairness metric computation:

  * Statistical Parity Difference (SPD)
  * Equal Opportunity Difference (EOpp)

These modules are imported and used within the notebook.

---

## Folder Structure

```
.
├── data/                   # Created after running notebook
│   ├── civil/
│   └── jigsaw/
|── pyproject.toml
├── src/
│   ├── download_data.py
│   ├── fairness_metrics.py
│   └── llm_zero_shot_explain.py
├── zero_shot_hate_explanations.ipynb
└── README.md
```

---

## Resources

* Hugging Face Transformers Documentation
  [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)

* Integrated Gradients
  [https://arxiv.org/abs/1703.01365](https://arxiv.org/abs/1703.01365)

* Fairness in Machine Learning
  [https://fairmlbook.org/](https://fairmlbook.org/)

---

### Final Notes

* The notebook is the intended entry point.
* Script-based CLI execution is available but not required.
* The emphasis of this reference implementation is interpretability-driven bias analysis, not benchmark evaluation.

---