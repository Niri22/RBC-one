# Preference Alignment with DPO (LLM-as-a-Judge)

This folder contains a reference implementation of **Direct Preference Optimization (DPO)** using an **LLM-as-a-Judge** pipeline for dataset construction, preference learning, training, and evaluation.

## Contents

- **assets/** – Supporting resources such as figures, prompt templates, and example artifacts.
- **utils/** – Shared helper modules for dataset processing, prompt handling, DPO utilities, and evaluation logic.
- **01_dataset_construction.ipynb** – Constructs and preprocesses the base instruction–response dataset used for preference learning.
- **02_llm_as_judge_inference.ipynb** – Runs LLM-as-a-Judge inference to compare candidate responses and generate preference signals.
- **03_dpo_pair_construction.ipynb** – Converts judge outputs into (prompt, chosen, rejected) pairs compatible with DPO training.
- **04_dpo_training.ipynb** – Fine-tunes the base language model using Direct Preference Optimization (DPO).
- **05_evaluation.ipynb** – Evaluates and compares the base and DPO-aligned models using quantitative and qualitative metrics.

## Environment Setup

### Create Environment from Scratch

1. **Load required modules**
```bash
module load python/3.10.12
module load cuda-12.4
```

2. **Create and activate a virtual environment**
```bash
python -m venv dpo_env
source dpo_env/bin/activate
```

3. **Install PyTorch with CUDA 12.4 support**
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

4. **Install project dependencies**
```bash
pip install -r dpo_req.txt
```

5. **Install FlashAttention**
```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

<!-- 6. **Register the environment as a Jupyter kernel**
```bash
python -m ipykernel install --user --name dpo_env --display-name "dpo_env"
``` -->

## Notes

- Run notebooks sequentially from **01 → 05**.
- Ensure GPU availability before DPO training.
- The quality of alignment depends strongly on the judge model and prompt design.

