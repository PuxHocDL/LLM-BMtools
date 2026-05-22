"""Self-correction with execution feedback.

Generate code -> execute -> classify the error -> build directed feedback ->
regenerate, for at most a few rounds. See the slide "Self-Correction with
Execution Feedback".
"""

from .debug_loop import code_similarity, debug_loop
from .error_detector import ErrorInfo, ErrorLevel, detect_error
from .feedback_builder import build_feedback
from .prompts import build_code_prompt

__all__ = [
    "debug_loop",
    "code_similarity",
    "detect_error",
    "build_feedback",
    "build_code_prompt",
    "ErrorLevel",
    "ErrorInfo",
]
