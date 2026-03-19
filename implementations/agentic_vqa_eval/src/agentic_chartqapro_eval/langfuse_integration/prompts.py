"""Versioned prompt loading via Langfuse Prompt Management.

Usage:
    # Load prompt (falls back to file if Langfuse unavailable)
    text = get_prompt("planner_prompt", PLANNER_PROMPT_PATH)

    # Push current prompt files to Langfuse (run once before a new experiment)
    python -m agentic_chartqapro_eval.langfuse_integration.prompts
"""

import argparse
from pathlib import Path
from typing import Optional

from .client import get_client


# Prompt names as stored in Langfuse Prompt Management
PLANNER_PROMPT_NAME = "chartqapro_planner"
VISION_PROMPT_NAME = "chartqapro_vision"


def get_prompt(name: str, fallback_path: Path) -> str:
    """Return the latest versioned prompt from Langfuse, or read from file."""
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
    vision_path: Optional[Path] = None,
) -> None:
    """Upload current planner.txt and vision.txt to Langfuse Prompt Management."""
    client = get_client()
    if client is None:
        print("[langfuse] No client — skipping prompt push")
        return

    agents_dir = Path(__file__).parents[1] / "agents" / "prompts"
    planner_path = planner_path or (agents_dir / "planner.txt")
    vision_path = vision_path or (agents_dir / "vision.txt")

    for name, path in [
        (PLANNER_PROMPT_NAME, planner_path),
        (VISION_PROMPT_NAME, vision_path),
    ]:
        if not path.exists():
            print(f"[langfuse] Prompt file not found: {path}")
            continue
        text = path.read_text()
        try:
            client.create_prompt(name=name, prompt=text, type="text")
            print(f"[langfuse] Pushed prompt '{name}'")
        except Exception as exc:
            print(f"[langfuse] Failed to push prompt '{name}': {exc}")


def main() -> None:
    """Parse CLI arguments and push prompt files to Langfuse Prompt Management."""
    parser = argparse.ArgumentParser(description="Push prompt files to Langfuse Prompt Management")
    parser.add_argument("--planner", default=None, help="Path to planner.txt")
    parser.add_argument("--vision", default=None, help="Path to vision.txt")
    args = parser.parse_args()

    push_prompts(
        planner_path=Path(args.planner) if args.planner else None,
        vision_path=Path(args.vision) if args.vision else None,
    )


if __name__ == "__main__":
    main()
