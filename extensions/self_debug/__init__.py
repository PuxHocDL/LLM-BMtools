"""
Self-Debug Extension for Code Execution Errors

This module implements self-debug functionality for code generation errors.
When generated code fails to execute, the error traceback is used as feedback
to guide the model in fixing its own code.

Error Levels:
- Level 1: Hard Error (Python exception) - KeyError, IndexError, etc.
- Level 2: Empty/None output - Function returns empty when data exists
- Level 3: Format Mismatch - Output format doesn't match question requirements
"""

from .error_detector import ErrorLevel, ErrorInfo, detect_error
from .feedback_builder import build_feedback
from .debug_loop import debug_loop, code_similarity

__all__ = [
    "ErrorLevel",
    "ErrorInfo",
    "detect_error",
    "build_feedback",
    "debug_loop",
    "code_similarity",
]
