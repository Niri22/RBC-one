"""PlannerAgent — text-only CrewAI agent that produces a strict JSON inspection plan.

The planner never sees the image. It creates a 2-4 step chart-reading procedure
that the VisionAgent will follow.
"""

import os
from pathlib import Path
from typing import Any, Optional, Tuple

from crewai import LLM, Agent, Crew, Task

from ..datasets.perceived_sample import PerceivedSample
from ..langfuse_integration.tracing import close_span, open_llm_span
from ..utils.json_strict import parse_strict


PLANNER_PROMPT_PATH = Path(__file__).parent / "prompts" / "planner.txt"

PLAN_REQUIRED_KEYS = [
    "steps",
    "expected_answer_type",
    "question_type",
    "answerability_check",
    "hints",
]


def _load_template() -> str:
    """
    Read the planning instructions from the project's prompt configuration.

    Returns
    -------
    str
        The content of the planner prompt template.
    """
    return PLANNER_PROMPT_PATH.read_text()


def build_planner_prompt(sample: PerceivedSample) -> str:
    """
    Inject sample details into the specialized planning template.

    Parameters
    ----------
    sample : PerceivedSample
        The data sample containing the question and context.

    Returns
    -------
    str
        The rendered prompt for the planner agent.
    """
    template = _load_template()

    choices_block = ""
    if sample.choices:
        choices_block = f"Choices: {', '.join(sample.choices)}"

    context_block = ""
    if sample.context:
        lines = ["Conversation context:"]
        for turn in sample.context:
            lines.append(f"  {turn.get('role', 'user')}: {turn.get('content', '')}")
        context_block = "\n".join(lines)

    return template.format(
        question=sample.question,
        question_type=sample.question_type.value,
        choices_block=choices_block,
        context_block=context_block,
    )


def _build_llm(backend: str, model: str, api_key: Optional[str]) -> LLM:
    """
    Configure the model interface for the planner.

    Parameters
    ----------
    backend : {'openai', 'gemini'}
        The language model provider.
    model : str
        The specific model name.
    api_key : str, optional
        Key for accessing the model.

    Returns
    -------
    LLM
        The initialized model controller.
    """
    if backend == "openai":
        return LLM(
            model=model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            temperature=0,
        )
    if backend == "gemini":
        return LLM(
            model=f"gemini/{model}",
            api_key=api_key or os.environ.get("GEMINI_API_KEY", ""),
            temperature=0,
        )
    raise ValueError(f"Unknown planner backend: {backend!r}")


class PlannerAgent:
    """
    An agent responsible for deriving a logical inspection strategy.

    The `PlannerAgent` processes a chart question without visual input to
    prescribe a detailed procedure for a subsequent vision-capable agent.
    """

    def __init__(
        self,
        backend: str = "gemini",
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the planner with the desired backend and model.

        Parameters
        ----------
        backend : str, default 'gemini'
            The provider for the planning model.
        model : str, default 'gemini-2.5-flash-lite'
            The model name.
        api_key : str, optional
            API key for the provider.
        """
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self._llm = _build_llm(backend, model, api_key)

    def run(self, sample: PerceivedSample, lf_trace: Any = None) -> Tuple[str, dict, bool, str]:
        """
        Execute the planning phase for a new question.

        Uses CrewAI to orchestrate the generation and parsing of the
        instructional plan.

        Parameters
        ----------
        sample : PerceivedSample
            The question and context to plan for.
        langfuse_trace : Any, optional
            Observability object for logging.

        Returns
        -------
        prompt : str
            The task description given to the planner.
        parsed : dict
            The extracted plan (steps, question_type, etc.).
        parse_error : bool
            True if JSON parsing failed.
        raw_text : str
            The raw response from the LLM.
        """
        prompt = build_planner_prompt(sample)

        span = open_llm_span(
            lf_trace,
            name="planner",
            input_data={"prompt": prompt},
            model=self.model,
            metadata={"backend": self.backend},
        )

        agent = Agent(
            role="Chart Reading Planner",
            goal=(
                "Produce a precise JSON inspection plan for answering a question about a chart. "
                "Output JSON only — no extra text."
            ),
            backstory=(
                "You are an expert chart analyst. You plan structured chart-reading procedures "
                "without seeing the image, so a vision agent can follow your steps precisely."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )

        task = Task(
            description=prompt,
            expected_output=(
                "A JSON object with keys: steps (list of 2-4 strings), "
                "expected_answer_type, question_type, answerability_check, hints"
            ),
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()

        raw_text: str = getattr(result, "raw", None) or str(result)
        parsed, parse_ok = parse_strict(raw_text, required_keys=PLAN_REQUIRED_KEYS)

        # Enforce step count bounds
        if parsed and "steps" in parsed:
            steps = list(parsed["steps"])
            if len(steps) < 2:
                steps += ["Check answerability from the chart data"] * (2 - len(steps))
            parsed["steps"] = steps[:4]

        close_span(
            span,
            output={"plan_steps": parsed.get("steps", []), "parse_error": not parse_ok},
        )

        return prompt, parsed, not parse_ok, raw_text
