r"""Pass 4: failure taxonomy — classify WHY each wrong answer failed.

Unlike the LLM judge (which uses text only), this pass gives a VLM both the
chart image AND the agent's wrong answer so it can make a grounded, visual
diagnosis of the failure mode.

Usage:
    uv run --env-file .env -m agentic_chartqapro_eval.eval.error_taxonomy \\
        --mep_dir meps/openai_openai/chartqapro/test \\
        --metrics_file metrics.jsonl \\
        --out taxonomy.jsonl
"""

import argparse
import base64
import contextlib
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from openai import OpenAI

from ..langfuse_integration.client import get_client
from ..mep.writer import iter_meps
from ..utils.json_strict import parse_strict


load_dotenv()

# ---------------------------------------------------------------------------
# Taxonomy categories
# ---------------------------------------------------------------------------

TAXONOMY_CATEGORIES = [
    "axis_misread",  # read the wrong axis value, scale, or unit
    "legend_confusion",  # confused series, colours, or legend entries
    "arithmetic_mistake",  # correct data extracted but calculation is wrong
    "hallucinated_element",  # referenced data / labels NOT visible in the chart
    "unanswerable_failure",  # should say UNANSWERABLE but didn't, or vice versa
    "question_misunderstanding",  # answered a different / adjacent question
    "extraction_error",  # could not locate the relevant data in the chart
    "other",  # none of the above
]

_TAXONOMY_PROMPT = """\
You are an expert chart QA error analyst. You can see the chart image.

Question         : {question}
Correct Answer   : {expected}
Agent's Answer   : {predicted}   ← WRONG
Agent Explanation: {explanation}

Inspection plan the agent should have followed:
{plan_steps}

Looking at the chart image AND the agent's explanation, classify the PRIMARY
reason the agent got this wrong into exactly ONE of these categories:

  axis_misread              – read the wrong axis value, scale, or unit
  legend_confusion          – mixed up series, colours, or legend entries
  arithmetic_mistake        – extracted correct data but made a calculation error
  hallucinated_element      – referenced data/labels NOT visible in this chart
  unanswerable_failure      – should say UNANSWERABLE but didn't, or vice versa
  question_misunderstanding – answered a different/adjacent question
  extraction_error          – could not locate the relevant data in the chart at all
  other                     – none of the above

Output ONLY JSON, no markdown, no extra text:
{{"failure_type": "<category>", "failure_reason": "<one sentence grounded in what you see in the chart>"}}"""

_TAXONOMY_KEYS = ["failure_type", "failure_reason"]


# ---------------------------------------------------------------------------
# VLM helpers (multimodal — image + text)
# ---------------------------------------------------------------------------


def _encode_image(image_path: str) -> tuple:
    """Return (base64_string, mime_type) for an image file."""
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = {
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "png": "png",
        "gif": "gif",
        "webp": "webp",
    }.get(ext, "jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime


def _call_vlm_openai(prompt: str, image_path: str, model: str, api_key: Optional[str]) -> str:
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
    b64, mime = _encode_image(image_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        max_completion_tokens=256,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def _call_vlm_gemini(prompt: str, image_path: str, model: str, api_key: Optional[str]) -> str:
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
    b64, mime = _encode_image(image_path)
    response = client.models.generate_content(
        model=model,
        contents=[genai.types.Part.from_bytes(data=b64, mime_type=f"image/{mime}"), prompt],
        config=genai.types.GenerateContentConfig(temperature=0, max_output_tokens=256),
    )
    return response.text or ""


# ---------------------------------------------------------------------------
# Per-MEP classification
# ---------------------------------------------------------------------------


def classify_failure(
    mep: dict,
    answer_accuracy: float,
    backend: str = "gemini",
    model: str = "gemini-2.5-flash-lite",
    api_key: Optional[str] = None,
) -> dict:
    """Classify WHY the agent failed on this sample.

    Returns
    -------
        dict with keys ``failure_type`` and ``failure_reason``.
        If ``answer_accuracy == 1.0``, returns immediately with
        ``failure_type="correct"`` without making a VLM call.
    """
    if answer_accuracy >= 1.0:
        return {"failure_type": "correct", "failure_reason": ""}

    sample = mep.get("sample", {})
    plan = mep.get("plan", {}).get("parsed", {})
    vision = mep.get("vision", {}).get("parsed", {})

    question = sample.get("question", "")
    expected = sample.get("expected_output", "")
    predicted = vision.get("answer", "")
    explanation = vision.get("explanation", "")
    image_path = sample.get("image_ref", {}).get("path", "") or ""
    plan_steps = plan.get("steps", [])
    steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan_steps)) or "  (none)"

    prompt = _TAXONOMY_PROMPT.format(
        question=question or "(unknown)",
        expected=expected,
        predicted=predicted,
        explanation=explanation or "(no explanation)",
        plan_steps=steps_text,
    )

    # Skip if image is missing — fall back to text-only best-guess
    has_image = image_path and Path(image_path).exists()

    try:
        if has_image:
            if backend == "openai":
                raw = _call_vlm_openai(prompt, image_path, model, api_key)
            elif backend == "gemini":
                raw = _call_vlm_gemini(prompt, image_path, model, api_key)
            else:
                raise ValueError(f"Unknown backend: {backend!r}")
        # Image missing — use text-only fallback via the same models
        elif backend == "openai":
            client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt + "\n\n(Note: chart image unavailable)",
                    }
                ],
                temperature=0,
                max_completion_tokens=256,
            )
            raw = resp.choices[0].message.content or ""
        elif backend == "gemini":
            client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
            resp = client.models.generate_content(
                model=model,
                contents=[prompt + "\n\n(Note: chart image unavailable)"],
                config=genai.types.GenerateContentConfig(temperature=0, max_output_tokens=256),
            )
            raw = resp.text or ""
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        result, ok = parse_strict(raw, required_keys=_TAXONOMY_KEYS)
        if not result:
            return {
                "failure_type": "other",
                "failure_reason": raw[:200],
                "parse_error": True,
            }

        # Normalise category to valid set
        ft = result.get("failure_type", "other").strip().lower()
        if ft not in TAXONOMY_CATEGORIES:
            ft = "other"
        result["failure_type"] = ft
        return result

    except Exception as exc:
        return {"failure_type": "other", "failure_reason": f"Taxonomy error: {exc}"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0915
    """Classify failures in MEP files and write taxonomy results to JSONL."""
    parser = argparse.ArgumentParser(description="Pass 4: failure taxonomy via VLM")
    parser.add_argument("--mep_dir", required=True, help="Directory containing MEP JSON files")
    parser.add_argument(
        "--metrics_file",
        default=None,
        help="Optional metrics.jsonl from eval_outputs — used to look up answer_accuracy",
    )
    parser.add_argument("--out", default="taxonomy.jsonl", help="Output JSONL file")
    parser.add_argument("--backend", default="gemini", choices=["openai", "gemini"])
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument(
        "--all",
        dest="classify_all",
        action="store_true",
        help="Classify correct samples too (by default they are skipped)",
    )
    args = parser.parse_args()

    # Build accuracy lookup from metrics.jsonl if provided
    accuracy_by_id: dict = {}
    if args.metrics_file and Path(args.metrics_file).exists():
        with open(args.metrics_file) as f:
            for raw_line in f:
                line = raw_line.strip()
                if line:
                    row = json.loads(line)
                    accuracy_by_id[row.get("sample_id", "")] = row.get("answer_accuracy", 0.0)

    lf_client = get_client()

    with open(args.out, "w") as f_out:
        count = 0
        skipped = 0
        for mep in iter_meps(args.mep_dir):
            sample = mep.get("sample", {})
            sample_id = sample.get("sample_id", "")
            config_name = mep.get("config", {}).get("config_name", "")
            question_type = sample.get("question_type", "standard")
            expected = sample.get("expected_output", "")
            predicted = mep.get("vision", {}).get("parsed", {}).get("answer", "")

            # Get accuracy from lookup (preferred) or recompute simply
            answer_accuracy = accuracy_by_id.get(sample_id, -1.0)
            if answer_accuracy < 0:
                # fallback: 1.0 if strings match case-insensitively
                answer_accuracy = 1.0 if expected.strip().lower() == predicted.strip().lower() else 0.0

            if answer_accuracy >= 1.0 and not args.classify_all:
                skipped += 1
                row = {
                    "sample_id": sample_id,
                    "config_name": config_name,
                    "question_type": question_type,
                    "expected": expected,
                    "predicted": predicted,
                    "answer_accuracy": answer_accuracy,
                    "failure_type": "correct",
                    "failure_reason": "",
                }
                f_out.write(json.dumps(row) + "\n")
                count += 1
                continue

            try:
                result = classify_failure(mep, answer_accuracy, backend=args.backend, model=args.model)
                row = {
                    "sample_id": sample_id,
                    "config_name": config_name,
                    "question_type": question_type,
                    "expected": expected,
                    "predicted": predicted,
                    "answer_accuracy": answer_accuracy,
                    **result,
                }
                f_out.write(json.dumps(row) + "\n")
                count += 1

                # Log to Langfuse if trace_id is available
                lf_trace_id = mep.get("lf_trace_id")
                if lf_client and lf_trace_id:
                    failure_type = result.get("failure_type", "other")
                    with contextlib.suppress(Exception):
                        lf_client.create_score(
                            trace_id=lf_trace_id,
                            name=f"failure_{failure_type}",
                            value=1.0,
                        )

                if count % 10 == 0:
                    print(f"  classified {count} samples …")

            except Exception as exc:
                print(f"  Error on {sample_id}: {exc}")

    print(f"\nDone. {count} rows written to {args.out}  ({skipped} correct samples recorded as-is)")

    # Print a quick breakdown
    breakdown: dict = {}
    with open(args.out) as f:
        for line in f:
            ft = json.loads(line).get("failure_type", "?")
            breakdown[ft] = breakdown.get(ft, 0) + 1
    print("\nFailure type breakdown:")
    for ft, n in sorted(breakdown.items(), key=lambda x: -x[1]):
        print(f"  {ft:<30} {n}")


if __name__ == "__main__":
    main()
