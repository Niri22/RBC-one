"""Versioned prompt loading via Langfuse Prompt Management.

Registers the three active SQL Assistant prompts (planner, sql_retrieval,
sql_generator) in Langfuse so experiments link to exact prompt versions.

Usage:
    # Load prompt (falls back to local file if Langfuse unavailable)
    text = get_prompt("sql_assistant_planner", PLANNER_PROMPT_PATH)

    # Push current prompt files to Langfuse (run once before a new experiment)
    uv run -m agentic_chartqapro_eval.langfuse_integration.prompts
"""

import argparse
from pathlib import Path
from typing import Optional

from .client import get_client


_AGENTS_PROMPTS_DIR = Path(__file__).parents[1] / "agents" / "prompts"

# Prompt names as stored in Langfuse Prompt Management
PLANNER_PROMPT_NAME = "sql_assistant_planner"
SQL_RETRIEVAL_PROMPT_NAME = "sql_assistant_schema_retriever"
SQL_GENERATOR_PROMPT_NAME = "sql_assistant_sql_generator"

# Canonical local paths
PLANNER_PROMPT_PATH = _AGENTS_PROMPTS_DIR / "planner.txt"
SQL_RETRIEVAL_PROMPT_PATH = _AGENTS_PROMPTS_DIR / "sql_retrieval.txt"
SQL_GENERATOR_PROMPT_PATH = _AGENTS_PROMPTS_DIR / "sglgenerator.txt"


def get_prompt(name: str, fallback_path: Path) -> str:
    """Return the latest versioned prompt from Langfuse, or read from local file."""
    client = get_client()
    if client:
        try:
            prompt = client.get_prompt(name=name)
            if prompt:
                return prompt.compile()
        except Exception:
            pass
    return fallback_path.read_text()


def push_prompts(
    planner_path: Optional[Path] = None,
    sql_retrieval_path: Optional[Path] = None,
    sql_generator_path: Optional[Path] = None,
) -> None:
    """Upload the three SQL Assistant prompts to Langfuse Prompt Management."""
    client = get_client()
    if client is None:
        print("[langfuse] No client — skipping prompt push")
        return

    prompts = [
        (PLANNER_PROMPT_NAME, planner_path or PLANNER_PROMPT_PATH),
        (SQL_RETRIEVAL_PROMPT_NAME, sql_retrieval_path or SQL_RETRIEVAL_PROMPT_PATH),
        (SQL_GENERATOR_PROMPT_NAME, sql_generator_path or SQL_GENERATOR_PROMPT_PATH),
    ]

    for name, path in prompts:
        if not path.exists():
            print(f"[langfuse] Prompt file not found: {path}")
            continue
        text = path.read_text()
        try:
            client.create_prompt(name=name, prompt=text, type="text")
            print(f"[langfuse] Pushed prompt '{name}' from {path.name}")
        except Exception as exc:
            print(f"[langfuse] Failed to push prompt '{name}': {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push SQL Assistant prompts to Langfuse")
    parser.add_argument("--planner", default=None, help="Override path to planner.txt")
    parser.add_argument("--sql_retrieval", default=None, help="Override path to sql_retrieval.txt")
    parser.add_argument("--sql_generator", default=None, help="Override path to sglgenerator.txt")
    args = parser.parse_args()

    push_prompts(
        planner_path=Path(args.planner) if args.planner else None,
        sql_retrieval_path=Path(args.sql_retrieval) if args.sql_retrieval else None,
        sql_generator_path=Path(args.sql_generator) if args.sql_generator else None,
    )


if __name__ == "__main__":
    main()
