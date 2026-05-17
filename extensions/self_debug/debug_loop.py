"""
Debug Loop for Self-Debug

Main loop that iterates through debug rounds, executing code and
collecting feedback.
"""

import difflib
import traceback as tb_module
from typing import Any, Dict, List, Tuple, Callable, Optional

from .error_detector import ErrorLevel, ErrorInfo, detect_error
from .feedback_builder import build_feedback


def debug_loop(
    model_generate: Callable[[str, float], str],
    execute_code: Callable[[str, Any], Any],
    prompt_orig: str,
    api_response: Any,
    question: str,
    schema: str,
    max_rounds: int = 3,
    error_levels: Tuple[int, ...] = (1,),
    temperature: float = 0.3,
) -> Tuple[Any, List[Dict]]:
    """
    Run the debug loop.

    Args:
        model_generate: Function that generates code from prompt (prompt, temp) -> code
        execute_code: Function that executes code (code, api_response) -> output
        prompt_orig: Original prompt
        api_response: The API response JSON
        question: The question being answered
        schema: Schema for the API response
        max_rounds: Maximum number of debug rounds
        error_levels: Tuple of error levels to fix (1=hard, 2=empty, 3=format)
        temperature: Temperature for code generation (higher for debugging)

    Returns:
        Tuple of (final_output, history_list)
    """
    history = []

    # Round 0: Initial generation (greedy)
    code = model_generate(prompt_orig, temperature=0)

    for round_i in range(max_rounds + 1):
        # Execute code
        exc = None
        output = None

        try:
            output = execute_code(code, api_response)
            err = detect_error(code, output, None, question, schema)
        except Exception as e:
            exc = e
            err = detect_error(code, None, e, question, schema)
            output = None

        # Record this round
        history.append({
            "round": round_i,
            "code": code,
            "output": _serialize_output(output),
            "error": {
                "level": err.level.value,
                "message": err.message,
                "traceback": err.traceback,
            },
        })

        # Success!
        if err.level == ErrorLevel.OK:
            return output, history

        # Error type not in scope to fix
        if err.level.value not in error_levels:
            return output, history

        # Max rounds reached
        if round_i == max_rounds:
            break

        # Detect "stuck": code unchanged across rounds
        if round_i > 0:
            sim = code_similarity(history[-1]["code"], history[-2]["code"])
            if sim > 0.995:
                history[-1]["terminated"] = "stuck"
                return output, history

        # Build feedback and generate new code
        try:
            feedback = build_feedback(
                code,
                err,
                schema,
                _get_sample_record(api_response),
                question=question,
            )
            prompt_new = prompt_orig + "\n\n" + feedback
            code = model_generate(prompt_new, temperature)
        except Exception as e:
            # If feedback generation fails, abort
            history[-1]["terminated"] = f"feedback_error: {e}"
            return output, history

    history[-1]["terminated"] = "max_rounds"
    return output, history


def code_similarity(c1: str, c2: str) -> float:
    """
    Calculate similarity between two code strings.

    Returns 0.0 to 1.0, where 1.0 means identical.
    """
    return difflib.SequenceMatcher(None, c1, c2).ratio()


def _serialize_output(output: Any) -> Any:
    """Serialize output for JSON storage."""
    if output is None:
        return None
    if isinstance(output, (str, int, float, bool)):
        return output
    if isinstance(output, (list, dict)):
        return output
    return str(output)


def _get_sample_record(api_response: Any) -> str:
    """Extract a sample record from the API response for context."""
    try:
        import json

        # Handle nested dict structure
        if isinstance(api_response, dict):
            # Look for common patterns
            if "available" in api_response and isinstance(api_response["available"], list):
                if api_response["available"]:
                    return json.dumps(api_response["available"][0], indent=2)[:500]

            # Look for any list with items
            for v in api_response.values():
                if isinstance(v, list) and v:
                    return json.dumps(v[0], indent=2)[:500]
                if isinstance(v, dict):
                    return json.dumps(v, indent=2)[:500]

            return json.dumps(api_response, indent=2)[:500]

        return str(api_response)[:500]
    except Exception:
        return "N/A"
