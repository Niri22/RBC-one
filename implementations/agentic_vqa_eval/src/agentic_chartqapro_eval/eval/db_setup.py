"""db_setup.py — Load CSV into a local SQLite database for the eval run.

Run once before eval_runner.py, or call setup_db() from main().
"""

import sqlite3
import pandas as pd
from pathlib import Path


def setup_db(csv_path: str, db_path: str = "rbc_metrics.db") -> str:
    """
    Load the credit card CSV into a SQLite database.

    Parameters
    ----------
    csv_path : str
        Path to the UCI credit card CSV file.
    db_path : str
        Output SQLite file path.

    Returns
    -------
    str
        SQLAlchemy-compatible URI: 'sqlite:///<db_path>'
    """
    print(f"Loading {csv_path} into {db_path}...")

    p = Path(csv_path)
    if p.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(csv_path, header=1)  # header=1 skips the UCI double-header row
    else:
        df = pd.read_csv(csv_path, header=1)

    # Rename the target column to something readable
    df = df.rename(columns={
        "default payment next month": "default_payment_next_month"
    })

    # Drop the ID column — not useful for metrics
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    conn = sqlite3.connect(db_path)
    df.to_sql(
        name="credit_card_clients",
        con=conn,
        if_exists="replace",   # safe to re-run
        index=False,
    )
    conn.close()

    row_count = len(df)
    print(f"Done. {row_count} rows written to table 'credit_card_clients' in {db_path}")
    print(f"Columns: {list(df.columns)}")

    return f"sqlite:///{Path(db_path).resolve()}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to UCI credit card CSV")
    parser.add_argument("--db",  default="rbc_metrics.db")
    args = parser.parse_args()
    uri = setup_db(args.csv, args.db)
    print(f"\nDB URI for eval_runner: {uri}")