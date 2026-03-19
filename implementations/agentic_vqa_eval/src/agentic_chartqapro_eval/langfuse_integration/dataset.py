"""Register ChartQAPro samples as a Langfuse Dataset.

Usage:
    python -m agentic_chartqapro_eval.langfuse_integration.dataset \
        --split test --n 25
"""

import argparse
from typing import Optional

from ..datasets.chartqapro_loader import load_chartqapro
from .client import get_client


def register_dataset(
    samples,
    dataset_name: str = "ChartQAPro",
    split: str = "test",
) -> Optional[str]:
    """
    Upload a collection of samples as a Langfuse Dataset.

    Allows for versioned dataset management and evaluation in the Langfuse UI.

    Parameters
    ----------
    samples : list of PerceivedSample
        The data samples to register.
    dataset_name : str, default 'ChartQAPro'
        The base name for the dataset.
    split : str, default 'test'
        The split identifier (e.g., 'train', 'val').

    Returns
    -------
    str or None
        The name of the created dataset if successful, else None.
    """
    client = get_client()
    if client is None:
        return None

    name = f"{dataset_name}_{split}"
    try:
        client.create_dataset(name=name)
        [
            client.create_dataset_item(
                dataset_name=name,
                input={
                    "source_id": s.sample_id,  # stored as data field; Langfuse auto-generates UUID v7 id
                    "question": s.question,
                    "question_type": s.question_type.value,
                    "image_path": s.image_path or "",
                    "choices": s.choices or [],
                },
                expected_output=s.expected_output,
            )
            for s in samples
        ]
        print(f"[langfuse] Registered {len(samples)} samples → dataset '{name}'")
        return name
    except Exception as exc:
        print(f"[langfuse] Dataset registration failed: {exc}")
        return None


def main() -> None:
    """
    Command-line interface for registering ChartQAPro datasets.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Register ChartQAPro samples as Langfuse dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--image_dir", default="data/chartqapro_images")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    samples = load_chartqapro(split=args.split, n=args.n, image_dir=args.image_dir, cache_dir=args.cache_dir)
    register_dataset(samples, split=args.split)


if __name__ == "__main__":
    main()
