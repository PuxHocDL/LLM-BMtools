"""Sandboxed code execution shared by the code-generation methods.

The model writes a small Python script that has the (pruned) JSON object
available as the global variable ``data`` and prints the answer with
``print()``. ``run_code`` executes that script and returns whatever it
printed; any execution error propagates to the caller so the
self-correction loop can react to it.
"""

import contextlib
import io
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta


def extract_code(model_output):
    """Pull the Python source out of a model response.

    Handles ```python fenced blocks, bare ``` blocks, or raw code.
    """
    if not isinstance(model_output, str):
        return ""
    fenced = re.search(r"```(?:python)?\s*(.*?)```", model_output, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return model_output.strip()


def _exec_globals(data):
    """Execution environment: the JSON payload plus a few common stdlib names."""
    return {
        "__builtins__": __builtins__,
        "data": data,
        "json": json,
        "re": re,
        "math": math,
        "statistics": statistics,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "datetime": datetime,
        "date": date,
        "timedelta": timedelta,
    }


def run_code(code, data):
    """Execute ``code`` with ``data`` in scope and return whatever it printed.

    Raises the underlying exception on failure -- the self-correction loop
    depends on seeing the traceback.
    """
    if not code or not code.strip():
        raise ValueError("No executable code was produced")
    buffer = io.StringIO()
    env = _exec_globals(data)
    with contextlib.redirect_stdout(buffer):
        exec(code, env)
    return buffer.getvalue().strip()
