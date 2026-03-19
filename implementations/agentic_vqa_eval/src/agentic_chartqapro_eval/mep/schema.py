"""Model Evaluation Packet (MEP) schema — portable trace artifact.

MEP v1 stores everything needed to replay and audit a single agent run.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ImageRef:
    """Reference to an image used in the question, e.g. chart screenshot."""

    path: str
    sha256: str


@dataclass
class MEPConfig:
    """Configuration of the agent run, including backend and model choices."""

    planner_backend: str  # "openai" | "gemini"
    vision_backend: str
    judge_backend: str
    config_name: str  # e.g. "openai_gemini"
    planner_model: str
    vision_model: str


@dataclass
class MEPSample:
    """Metadata about the question sample.

    Includes dataset, question text, and image reference.
    """

    dataset: str
    sample_id: str
    question: str
    question_type: str
    expected_output: str
    image_ref: ImageRef
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MEPPlan:
    """The agent's plan for answering the question.

    Includes the original prompt, raw response text, and parsed plan.
    """

    prompt: str
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)
    parse_error: bool = False


@dataclass
class ToolTrace:
    """Trace of a single tool call made by the agent.

    Includes the tool name, backend, model, timestamps, and any
    provider-specific metadata.
    """

    tool: str
    backend: str
    model: str
    start_ts: str
    end_ts: str
    elapsed_ms: float = 0.0
    provider_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MEPVision:
    """The agent's vision step.

    Includes the original prompt, raw response text, and parsed vision
    output (e.g. answer and explanation).
    """

    prompt: str
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)  # {answer, explanation}
    parse_error: bool = False
    tool_trace: List[Dict] = field(default_factory=list)


@dataclass
class MEPOcr:
    """The agent's OCR step.

    Includes the raw response text and parsed OCR output (e.g. chart type,
    title, axes labels, legend, data labels, annotations).
    """

    raw_text: str
    parsed: Dict[str, Any] = field(
        default_factory=dict
    )  # {chart_type, title, x_axis, y_axis, legend, data_labels, annotations}
    parse_error: bool = False
    tool_trace: List[Dict] = field(default_factory=list)


@dataclass
class MEPVerifier:
    """The agent's verifier step.

    Includes the original prompt, raw response text, parsed verifier output
    (e.g. verdict, revised answer, and reasoning), and final verdict
    (confirmed, revised, or skipped).
    """

    prompt: str
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)  # {verdict, answer, reasoning}
    parse_error: bool = False
    verdict: str = "skipped"  # "confirmed" | "revised" | "skipped"


@dataclass
class MEPTimestamps:
    """Timestamps for the different steps of the agent run.

    Includes the start and end time of the entire run, as well as the time
    taken for each individual step (planner, OCR, vision, verifier).
    """

    start: str
    end: str
    planner_ms: float = 0.0
    ocr_ms: float = 0.0  # 0.0 when OCR step is skipped
    vision_ms: float = 0.0
    verifier_ms: float = 0.0


@dataclass
class MEP:
    """The complete Model Evaluation Packet for a single agent run.

    Contains all relevant information for replay and audit, including schema
    version, run ID, configuration, sample metadata, planner output, OCR
    output (if applicable), vision output, verifier output (if applicable),
    timestamps, and any errors encountered during the run.
    """

    schema_version: str = "mep.v1"
    run_id: str = ""
    config: Optional[MEPConfig] = None
    sample: Optional[MEPSample] = None
    plan: Optional[MEPPlan] = None
    ocr: Optional[MEPOcr] = None  # None when OCR step is skipped
    vision: Optional[MEPVision] = None
    verifier: Optional[MEPVerifier] = None  # Pass 2.5 — None when skipped
    timestamps: Optional[MEPTimestamps] = None
    errors: List[str] = field(default_factory=list)
    lf_trace_id: Optional[str] = None  # set when Langfuse tracing is active

    def to_dict(self) -> dict:
        """Return a dict representation suitable for JSON serialization."""
        return dataclasses.asdict(self)
