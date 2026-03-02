#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download and normalize social-problem datasets into Parquet files.

Supported datasets:
- CivilComments (HF: google/civil_comments)

Examples of usage:
1. Stream and take 200,000 rows from the CivilComments dataset:
   python scripts/download_data.py --dataset civil
       --out data/civil.parquet --stream --take 200000

2. Stream and take 500 rows from the CivilComments dataset:
   uv run scripts/download_data.py --dataset civil
       --out data/civil.parquet --stream --take 500
"""

# Importing necessary libraries
import argparse  # For parsing command-line arguments
import itertools
import random
from collections.abc import Generator
from pathlib import Path  # For working with file paths
from typing import Any, Iterator, Optional

import pandas as pd  # For data manipulation and analysis
from datasets import (  # For loading datasets from Hugging Face
    load_dataset,
)


# Utility function to convert a value to a boolean
def as_bool(x: Any) -> bool:
    """Convert a value to a boolean."""
    return str(x).lower() in {"1", "true", "t", "yes", "y"}


# Function to save a DataFrame to a Parquet file
def to_parquet(df: pd.DataFrame, out_path: str) -> None:
    """Save a DataFrame to a Parquet file."""
    out_path_obj = Path(out_path)  # Convert the output path to a Path object
    out_path_obj.parent.mkdir(
        parents=True, exist_ok=True
    )  # Create parent directories if they don't exist
    df.to_parquet(out_path_obj, index=False)  # Save the DataFrame as a Parquet file
    print(
        f"Wrote {len(df):,} rows -> {out_path_obj}"
    )  # Print the number of rows written


# Function to load the CivilComments dataset
def load_civil(
    stream: bool = False, take: int | None = None, split: str = "train"
) -> pd.DataFrame:
    """Load the CivilComments dataset."""
    ds = load_dataset(
        "google/civil_comments", split=split, streaming=stream, trust_remote_code=True
    )
    cols_keep = [
        "text",
        "toxicity",
        "severe_toxicity",
        "obscene",
        "identity_attack",
        "insult",
        "threat",
    ]
    if stream:
        if take is None:
            raise ValueError(
                "When using --stream, you must specify --take to limit the number of rows. "
                "Example: --stream --take 200000"
            )
    
        it_gen: Generator[dict[str, Any], None, None] = (
            {
                "comment_text": r.get("text", ""),
                "target": r.get("toxicity", None),
                "severe_toxicity": r.get("severe_toxicity", None),
                "obscene": r.get("obscene", None),
                "identity_attack": r.get("identity_attack", None),
                "insult": r.get("insult", None),
                "threat": r.get("threat", None),
            }
            for r in ds
        )
    
        it = list(itertools.islice(it_gen, take))
        return pd.DataFrame(it)
    # Remove unwanted columns and rename for consistency
    ds = ds.remove_columns([c for c in ds.column_names if c not in cols_keep])
    return ds.to_pandas().rename(columns={"text": "comment_text", "toxicity": "target"})


# Main function to handle command-line arguments and process datasets
def main() -> None:
    """Handle command-line arguments and process datasets."""
    ap = argparse.ArgumentParser()  # Create an argument parser
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["civil"],
    )
    ap.add_argument("--out", required=True)  # Output file path
    ap.add_argument(
        "--stream", action="store_true", help="Use streaming mode (good for huge data)."
    )
    ap.add_argument(
        "--take",
        type=int,
        default=None,
        help="With --stream, cap the number of rows to read.",
    )
    ap.add_argument(
        "--sample", type=int, default=None, help="Downsample to N rows after load."
    )
    args = ap.parse_args()  # Parse command-line arguments

    # Load the appropriate dataset based on the argument
    if args.dataset == "civil":
        df = load_civil(stream=args.stream, take=args.take)
    else:
        raise ValueError("unknown dataset")

    # Downsample the dataset if requested
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)

    # Save the dataset to a Parquet file
    to_parquet(df, args.out)


# Entry point of the script
if __name__ == "__main__":
    main()
