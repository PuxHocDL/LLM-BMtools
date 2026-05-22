"""Error detection for self-correction.

Classifies the outcome of a code execution into one of four levels:
OK, hard error (exception), empty output, or format mismatch.
"""

import re
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ErrorLevel(Enum):
    """Classification of error types."""
    OK = 0
    HARD_ERROR = 1  # Python exception
    EMPTY = 2  # Empty/None output when data exists
    FORMAT_MISMATCH = 3  # Output format doesn't match question


@dataclass
class ErrorInfo:
    """Information about an error."""
    level: ErrorLevel
    message: str
    traceback: Optional[str] = None
    raw_output: Any = None


def detect_error(code: str, output: Any, exception: Optional[Exception],
                 question: str, schema: str) -> ErrorInfo:
    """
    Detect and classify errors from code execution.

    Args:
        code: The executed Python code
        output: The output from code execution (if successful)
        exception: The exception raised (if any)
        question: The original question/prompt
        schema: The schema for the API response

    Returns:
        ErrorInfo object with error details
    """
    # Level 1: Python exception (hard error)
    if exception is not None:
        return ErrorInfo(
            level=ErrorLevel.HARD_ERROR,
            message=str(exception),
            traceback=_format_tb(exception),
        )

    # Level 2: Empty/None output
    if is_empty_output(output) and not question_expects_empty(question):
        return ErrorInfo(
            level=ErrorLevel.EMPTY,
            message="Output is empty/None",
            raw_output=output,
        )

    # Level 3: Format mismatch
    fmt_err = check_format(output, question)
    if fmt_err:
        return ErrorInfo(
            level=ErrorLevel.FORMAT_MISMATCH,
            message=fmt_err,
            raw_output=output,
        )

    return ErrorInfo(level=ErrorLevel.OK, message="")


def is_empty_output(output: Any) -> bool:
    """Check if output is effectively empty."""
    if output is None:
        return True
    if output == "None":
        return True
    if isinstance(output, str):
        stripped = output.strip()
        if stripped in ("", "null", "n/a", "N/A", "none", "None"):
            return True
    if isinstance(output, (list, dict)) and len(output) == 0:
        return True
    return False


def question_expects_empty(question: str) -> bool:
    """
    Check if question expects an empty/None answer.

    Examples: "if any", "check if there are", "none if no"
    """
    patterns = [
        r"\bif any\b",
        r"\bnone if no\b",
        r"\bcheck if\b",
        r"\bwhether there (?:are|is)\b",
        r"\bverify if\b",
    ]
    q_lower = question.lower()
    return any(re.search(p, q_lower) for p in patterns)


def check_format(output: Any, question: str) -> Optional[str]:
    """
    Check if output format matches question requirements.

    Returns error message if mismatch, None if OK.
    """
    q_lower = question.lower()

    # Comma-separated list expected
    if "comma separated" in q_lower or "comma-separated" in q_lower:
        if not isinstance(output, str):
            return f"Expected comma-separated string, got {type(output).__name__}"

    # Number/count expected
    if ("how many" in q_lower or "count" in q_lower or
        "total number" in q_lower or "number of" in q_lower):
        try:
            val = str(output).strip()
            int(val)
        except (ValueError, TypeError):
            return f"Expected integer count, got: {output}"

    # Area/size expected (number)
    if "area" in q_lower or "size" in q_lower or "surface" in q_lower:
        try:
            val = str(output).strip()
            # Allow "322.9" or "322.9 sq ft" etc
            float(re.sub(r'[^\d.]', '', val))
        except (ValueError, TypeError):
            # If it fails, might still be OK (e.g., "Not specified")
            pass

    return None


def _format_tb(exception: Exception) -> str:
    """Format exception traceback."""
    tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
    return "".join(tb_lines).strip()
