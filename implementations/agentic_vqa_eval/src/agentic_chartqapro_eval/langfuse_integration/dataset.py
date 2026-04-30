"""Register SQL Assistant eval samples as a Langfuse Dataset.

Reads from eval_samples.json (the flat JSON eval set) and uploads each sample
as a Langfuse Dataset item so experiments can be linked to specific dataset
versions in the Langfuse UI.

Usage:
    uv run -m agentic_chartqapro_eval.langfuse_integration.dataset \
        --samples src/agentic_chartqapro_eval/eval/eval_samples.json \
        --name sql_assistant_eval
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from .client import get_client


def register_dataset(
    samples: List[dict],
    dataset_name: str = "sql_assistant_eval",
) -> Optional[str]:
    """Upload eval_samples.json entries as a Langfuse Dataset.

    Parameters
    ----------
    samples : list of dict
        Dicts with keys: sample_id, question, question_type, expected_output,
        kpi_name, metadata.
    dataset_name : str
        Name of the dataset in Langfuse.

    Returns
    -------
    str or None
        The dataset name if successful, else None.
    """
    client = get_client()
    if client is None:
        print("[langfuse] No client — skipping dataset registration")
        return None

    try:
        client.create_dataset(name=dataset_name)
        for s in samples:
            client.create_dataset_item(
                dataset_name=dataset_name,
                input={
                    "source_id": s.get("sample_id", ""),
                    "question": s.get("question", ""),
                    "question_type": s.get("question_type", "standard"),
                    "kpi_name": s.get("kpi_name", ""),
                    "metadata": s.get("metadata", {}),
                },
                expected_output=s.get("expected_output", ""),
            )
        print(f"[langfuse] Registered {len(samples)} samples → dataset '{dataset_name}'")
        return dataset_name
    except Exception as exc:
        print(f"[langfuse] Dataset registration failed: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Register SQL Assistant eval samples as Langfuse dataset")
    parser.add_argument(
        "--samples",
        default=str(Path(__file__).parents[1] / "eval" / "eval_samples.json"),
        help="Path to eval_samples.json",
    )
    parser.add_argument("--name", default="sql_assistant_eval", help="Dataset name in Langfuse")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N samples")
    args = parser.parse_args()

    with open(args.samples) as f:
        samples = json.load(f)

    if args.n is not None:
        samples = samples[: args.n]

    register_dataset(samples, dataset_name=args.name)


if __name__ == "__main__":
    main()
