"""Unified dataset-agnostic sample representation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


UNANSWERABLE_TOKEN = "UNANSWERABLE"

# Answer strings from the dataset that map to UNANSWERABLE
UNANSWERABLE_ANSWERS = {
    "unanswerable",
    "cannot be determined",
    "not determinable",
    "not answerable",
    "n/a",
    "none",
    "no answer",
}


class QuestionType(str, Enum):
    """
    Internal taxonomy for different vision-language task categories.

    Used for routing logic and evaluation grouping.
    """

    STANDARD = "standard"
    MCQ = "mcq"
    CONVERSATIONAL = "conversational"
    HYPOTHETICAL = "hypothetical"
    UNANSWERABLE = "unanswerable"
    # SQL Assistant pipeline question types
    EXEC = "exec"
    ANALYST = "analyst"


@dataclass
class PerceivedSample:
    """
    Standardized container for vision-language data points.

    Attributes
    ----------
    sample_id : str
        Unique identifier.
    image_path : str
        Disk path to the associated chart image.
    question : str
        The user text prompt.
    expected_output : str
        The ground-truth answer.
    question_type : QuestionType
        The task category.
    choices : list of str, optional
        Possible MCQ options.
    context : list of dict, optional
        Multimodal conversation history.
    metadata : dict
        Additional unstructured data fields.
    """

    sample_id: str
    image_path: str  # local path to saved image
    question: str
    expected_output: str  # canonical; UNANSWERABLE_TOKEN for unanswerable
    question_type: QuestionType
    choices: Optional[List[str]] = None  # MCQ option texts
    context: Optional[List[dict]] = None  # [{"role": "user"|"assistant", "content": "..."}]
    metadata: dict = field(default_factory=dict)

    def is_unanswerable(self) -> bool:
        """
        Check if the ground truth indicates the image is unanswerable.

        Returns
        -------
        bool
            True if the 'expected_output' is the canonical 'UNANSWERABLE' token.
        """
        return self.expected_output.strip().upper() == UNANSWERABLE_TOKEN

    def to_dict(self) -> dict:
        """
        Serialize the sample data into a standard Python dictionary.

        Returns
        -------
        dict
            The sample fields as key-value pairs.
        """
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "question": self.question,
            "expected_output": self.expected_output,
            "question_type": self.question_type.value,
            "choices": self.choices,
            "context": self.context,
            "metadata": self.metadata,
        }
