"""ChartQAPro dataset loader → PerceivedSample list.

Dataset: ahmed-masry/ChartQAPro
Columns:
  Question      list[str]   – one or more questions (conversational multi-turn)
  Answer        list[str]   – corresponding answers
  Question Type str         – e.g. "Factoid", "Unanswerable", "Multiple Choice", etc.
  image         PIL Image   – chart image
  Year          list[str]   – year context per turn (may be "NO")
  Paragraph     str         – caption/context paragraph
"""

import re
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from datasets import load_dataset
from PIL import Image as PILImage

from .perceived_sample import (
    UNANSWERABLE_ANSWERS,
    UNANSWERABLE_TOKEN,
    PerceivedSample,
    QuestionType,
)


# Map raw dataset question types to our canonical types
_TYPE_MAP = {
    "factoid": QuestionType.STANDARD,
    "unanswerable": QuestionType.UNANSWERABLE,
    "multiple choice": QuestionType.MCQ,
    "multiple-choice": QuestionType.MCQ,
    "mcq": QuestionType.MCQ,
    "conversational": QuestionType.CONVERSATIONAL,
    "hypothetical": QuestionType.HYPOTHETICAL,
    "hypothetical reasoning": QuestionType.HYPOTHETICAL,
}


def _map_qtype(raw: str) -> QuestionType:
    """
    Translate raw dataset category strings to internal QuestionType.

    Parameters
    ----------
    raw : str
        The raw label from the dataset.

    Returns
    -------
    QuestionType
        The corresponding internal category.
    """
    return _TYPE_MAP.get(raw.strip().lower(), QuestionType.STANDARD)


def _normalize_answer(answer: str, qtype: QuestionType) -> str:
    """
    Clean and standardize answer strings.

    Parameters
    ----------
    answer : str
        The raw answer from the dataset.
    qtype : QuestionType
        The category of the question.

    Returns
    -------
    str
        The normalized answer, using 'UNANSWERABLE' if appropriate.
    """
    if qtype == QuestionType.UNANSWERABLE:
        return UNANSWERABLE_TOKEN
    if answer.strip().lower() in UNANSWERABLE_ANSWERS:
        return UNANSWERABLE_TOKEN
    return answer.strip()


def _save_image(image, idx: int, img_dir: Path) -> str:
    """
    Export a PIL Image or binary image data to a local file.

    Parameters
    ----------
    image : PIL.Image, bytes, or dict
        The image data source.
    idx : int
        The sequential index for the filename.
    img_dir : Path
        The destination folder.

    Returns
    -------
    str
        The absolute filesystem path to the saved PNG.
    """
    path = img_dir / f"chart_{idx:06d}.png"
    if path.exists():
        return str(path)
    try:
        if isinstance(image, PILImage.Image):
            image.save(str(path))
        elif isinstance(image, bytes):
            with open(str(path), "wb") as f:
                f.write(image)
        elif isinstance(image, dict):
            # HuggingFace wraps as {"bytes": b"...", "path": "..."}
            raw = image.get("bytes") or image.get("path")
            if isinstance(raw, bytes):
                with open(str(path), "wb") as f:
                    f.write(raw)
            elif isinstance(raw, str) and raw:
                shutil.copy(raw, str(path))
            else:
                raise ValueError(f"Unknown image dict keys: {list(image.keys())}")
        else:
            PILImage.fromarray(np.array(image)).save(str(path))
        return str(path)
    except Exception as e:
        print(f"  Warning: could not save image row {idx}: {e}")
        return ""


def _extract_mcq_choices(question: str) -> Optional[List[str]]:
    """
    Parse multiple-choice options from within a natural language prompt.

    Searches specifically for labels like 'A)', 'B:', or 'C '.

    Parameters
    ----------
    question : str
        The question text.

    Returns
    -------
    list of str, optional
        The extracted choice labels, if found.
    """
    choices = re.findall(r"(?:^|(?<=\s))([A-D])[).:\s]+([^A-D\n]{3,})", question)
    if choices:
        return [c[1].strip() for c in choices]
    return None


def _normalize_row(row_idx: int, row: dict, img_dir: Path) -> List[PerceivedSample]:
    """
    Construct standardized sample objects from a raw dataset row.

    Handles conversational turns by creating multiple samples if necessary.

    Parameters
    ----------
    row_idx : int
        The row index from the source.
    row : dict
        The raw dictionary results from HuggingFace.
    img_dir : Path
        Folder for saving images.

    Returns
    -------
    list of PerceivedSample
        The transformed samples.
    """
    questions = row.get("Question") or []
    answers = row.get("Answer") or []
    qtype_raw = row.get("Question Type") or "Factoid"
    image = row.get("image")
    years = row.get("Year") or []
    paragraph = row.get("Paragraph") or ""

    qtype = _map_qtype(qtype_raw)

    # Save image once per row
    image_path = ""
    if image is not None:
        image_path = _save_image(image, row_idx, img_dir)

    if not questions:
        return []

    def _make_sample(turn_idx: int, prev_turns: list) -> PerceivedSample:
        question = questions[turn_idx]
        answer = answers[turn_idx] if turn_idx < len(answers) else ""
        answer = _normalize_answer(answer, qtype)
        year = years[turn_idx] if turn_idx < len(years) else None

        choices = _extract_mcq_choices(question) if qtype == QuestionType.MCQ else None

        context = None
        if prev_turns:
            context = []
            for pi in range(0, len(prev_turns), 2):
                context.append({"role": "user", "content": prev_turns[pi]})
                if pi + 1 < len(prev_turns):
                    context.append({"role": "assistant", "content": prev_turns[pi + 1]})

        sample_id = f"chartqapro_{row_idx:06d}" if len(questions) == 1 else f"chartqapro_{row_idx:06d}_t{turn_idx}"

        return PerceivedSample(
            sample_id=sample_id,
            image_path=image_path,
            question=question,
            expected_output=answer,
            question_type=qtype,
            choices=choices,
            context=context,
            metadata={
                "dataset": "ChartQAPro",
                "question_type_raw": qtype_raw,
                "year": year,
                "paragraph": paragraph,
                "row_idx": row_idx,
                "turn_idx": turn_idx,
                "n_turns": len(questions),
            },
        )

    if qtype != QuestionType.CONVERSATIONAL or len(questions) == 1:
        # Single-turn or last-turn of non-conversational multi-question
        turn_idx = len(questions) - 1
        prev = []
        for i in range(turn_idx):
            prev.append(questions[i])
            if i < len(answers):
                prev.append(answers[i])
        return [_make_sample(turn_idx, prev)]
    # Conversational: one sample per turn
    samples = []
    prev = []
    for turn_idx in range(len(questions)):
        samples.append(_make_sample(turn_idx, list(prev)))
        prev.append(questions[turn_idx])
        if turn_idx < len(answers):
            prev.append(answers[turn_idx])
    return samples


def load_chartqapro(
    split: str = "test",
    n: Optional[int] = None,
    image_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> List[PerceivedSample]:
    """
    High-level loader for the ChartQAPro dataset.

    Parameters
    ----------
    split : str, default 'test'
        The split to load.
    n : int, optional
        Maximum number of samples to return.
    image_dir : str, optional
        Where to save the extracted images.
    cache_dir : str, optional
        HuggingFace cache location.
    hf_token : str, optional
        API token for private data.

    Returns
    -------
    list of PerceivedSample
        The ready-to-use data records.
    """
    if image_dir is None:
        image_dir = "/home/coder/interpretability-llms-agents/data"
    img_dir = Path(image_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if hf_token:
        kwargs["token"] = hf_token

    print(f"Loading default of credit card clients split={split} …")
    file_path = r"/home/coder/interpretability-llms-agents/data/default of credit card clients.xls"
    # df = pd.read_excel(file_path)
    ds = load_dataset("xls", data_files=file_path)
    # ds = load_dataset(file_path, split=split, **kwargs)

    samples: List[PerceivedSample] = []
    for row_idx, row in enumerate(ds):
        row_samples = _normalize_row(row_idx, row, img_dir)
        samples.extend(row_samples)
        if n is not None and len(samples) >= n:
            samples = samples[:n]
            break

    print(f"  → {len(samples)} samples loaded (from {row_idx + 1} rows)")
    return samples
