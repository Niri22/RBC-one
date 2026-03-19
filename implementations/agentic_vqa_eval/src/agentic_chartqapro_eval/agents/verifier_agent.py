"""VerifierAgent — Pass 2.5: critically reviews the VisionAgent's draft answer.

The verifier sees the chart image AND the draft answer/explanation and decides
whether to CONFIRM or REVISE the answer. This teaches multi-agent critique patterns.

Unlike VisionAgent (which uses CrewAI + tool-use to explore the chart), the
verifier makes a single direct VLM call — showing that multi-agent critique
does not always require a full orchestration framework.

Key teaching point
------------------
Two agents looking at the same chart image can disagree. When the second model
(the verifier) has explicit access to the first model's reasoning, it can catch
errors the first model missed — axis misreads, arithmetic mistakes, etc.
"""

import base64
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from google import genai
from openai import OpenAI
from PIL import Image

from ..langfuse_integration.tracing import close_span, open_llm_span
from ..utils.json_strict import parse_strict


VERIFIER_REQUIRED_KEYS = ["verdict", "answer", "reasoning"]

_VERIFIER_PROMPT = """\
You are a critical chart QA verifier. A vision agent has already attempted to answer
the question below. Your job: look at the chart image carefully and audit the work.

Question         : {question}
Question Type    : {question_type}

Inspection plan the agent was supposed to follow:
{plan_steps}

Vision Agent's Draft Answer     : {draft_answer}
Vision Agent's Draft Explanation: {draft_explanation}

Examine the chart image. Then decide:
  CONFIRM — the draft answer is correct (output the same answer unchanged)
  REVISE  — you can see a clear, specific error; output the corrected answer

Rules:
- Only REVISE when you are confident you can point to a concrete error in the chart
- If uncertain, CONFIRM — do not second-guess without visual evidence
- For MCQ questions: the answer must be one of the stated choices
- If the answer is truly unanswerable from the chart, say exactly "UNANSWERABLE"
- Keep answers concise — numbers, short phrases, or single words where appropriate

Output ONLY JSON, no markdown, no extra text:
{{"verdict": "confirmed" | "revised", "answer": "<final answer>", "reasoning": "<one sentence grounded in what you see in the chart>"}}"""


# ---------------------------------------------------------------------------
# VLM helpers (same pattern as error_taxonomy.py)
# ---------------------------------------------------------------------------


def _encode_image(image_path: str) -> tuple:
    """
    Read an image file and convert it to a base64 string.

    Parameters
    ----------
    image_path : str
        The path to the image file on disk.

    Returns
    -------
    b64 : str
        The base64 encoded image string.
    mime : str
        The inferred MIME type of the image.
    """
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
    """
    Submit a vision request to the OpenAI API.

    Parameters
    ----------
    prompt : str
        The text instruction for the model.
    image_path : str
        Path to the relevant image.
    model : str
        The GPT model name.
    api_key : str, optional
        API key for authorization.

    Returns
    -------
    str
        The text response from the OpenAI model.
    """
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
    """
    Submit a vision request to the Google Gemini API.

    Parameters
    ----------
    prompt : str
        The text instruction for the model.
    image_path : str
        Path to the relevant image.
    model : str
        The Gemini model name.
    api_key : str, optional
        API key for authorization.

    Returns
    -------
    str
        The text response from the Gemini model.
    """
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
    image = Image.open(image_path)
    response = client.models.generate_content(
        model=model,
        contents=[image, prompt],
        config=genai.types.GenerateContentConfig(temperature=0, max_output_tokens=256),
    )
    return response.text or ""


# ---------------------------------------------------------------------------
# VerifierAgent
# ---------------------------------------------------------------------------


class VerifierAgent:
    """
    A validation agent that critiques draft answers against visual evidence.

    Coordinates a critical second look at the chart image to verify the
    logical grounding of the initial answer proposed by the VisionAgent.
    """

    def __init__(
        self,
        backend: str = "gemini",
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the verifier with the specified multimodal backend.

        Parameters
        ----------
        backend : {'openai', 'gemini'}
            The provider for the verification model.
        model : str
            The model name.
        api_key : str, optional
            API key to use for calls.
        """
        self.backend = backend
        self.model = model
        self.api_key = api_key

    def run(
        self,
        sample,  # PerceivedSample
        plan: dict,
        vision_parsed: dict,
        lf_trace: Any = None,
    ) -> Tuple[str, dict, bool, str]:
        """
        Critically audit a draft answer using a single VLM call.

        Parameters
        ----------
        sample : PerceivedSample
            The source data sample.
        plan : dict
            The inspection plan used by the previous agent.
        vision_parsed : dict
            The draft answer and explanation to audit.
        langfuse_trace : Any, optional
            Tracing object for observability.

        Returns
        -------
        prompt : str
            The verifier prompt rendered for the model.
        parsed : dict
            The final audited response containing 'verdict', 'answer', and 'reasoning'.
        parse_error : bool
            True if the JSON result was malformed.
        raw_text : str
            The raw string response from the VLM.
        """
        plan_steps = plan.get("steps", [])
        steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan_steps)) or "  (none)"

        draft_answer = vision_parsed.get("answer", "(none)")
        draft_explanation = vision_parsed.get("explanation", "(none)")
        question_type = getattr(
            getattr(sample, "question_type", None),
            "value",
            str(getattr(sample, "question_type", "standard")),
        )

        prompt = _VERIFIER_PROMPT.format(
            question=sample.question,
            question_type=question_type,
            plan_steps=steps_text,
            draft_answer=draft_answer,
            draft_explanation=draft_explanation,
        )

        span = open_llm_span(
            lf_trace,
            name="verifier",
            input_data={"prompt": prompt, "draft_answer": draft_answer},
            model=self.model,
            metadata={"backend": self.backend},
        )

        image_path = getattr(sample, "image_path", "") or ""
        has_image = image_path and Path(image_path).exists()

        try:
            if has_image:
                if self.backend == "openai":
                    raw = _call_vlm_openai(prompt, image_path, self.model, self.api_key)
                elif self.backend == "gemini":
                    raw = _call_vlm_gemini(prompt, image_path, self.model, self.api_key)
                else:
                    raise ValueError(f"Unknown backend: {self.backend!r}")
            else:
                # Image missing — cannot verify visually; default to confirm
                raw = (
                    '{"verdict": "confirmed", "answer": "'
                    + draft_answer.replace('"', '\\"')
                    + '", "reasoning": "Image unavailable; cannot verify visually."}'
                )

            parsed, parse_ok = parse_strict(raw, required_keys=VERIFIER_REQUIRED_KEYS)
            if not parsed:
                parsed = {
                    "verdict": "confirmed",
                    "answer": draft_answer,
                    "reasoning": f"Parse error — defaulting to confirm. Raw: {raw[:120]}",
                }
                parse_ok = False

            # Normalise verdict to known values
            if parsed.get("verdict", "").lower() not in ("confirmed", "revised"):
                parsed["verdict"] = "confirmed"

            close_span(span, output=parsed)
            return prompt, parsed, not parse_ok, raw

        except Exception as exc:
            fallback = {
                "verdict": "confirmed",
                "answer": draft_answer,
                "reasoning": f"Verifier error: {exc}",
            }
            close_span(span, output=fallback, error=str(exc))
            return prompt, fallback, True, ""
