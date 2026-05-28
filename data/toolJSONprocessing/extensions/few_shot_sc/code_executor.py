import ast
import json
import re
import math
import logging
from collections import Counter, defaultdict
from datetime import datetime, date, timedelta
from typing import Any

logger = logging.getLogger(__name__)


def _build_exec_globals(data_obj: Any) -> dict:
    """Pre-populate execution env: 'data' global + common stdlib modules.

    Why: model often references 'data' as a module-level var (prompt has `data = {...}`)
    or uses json/re/math/datetime without importing.
    """
    return {
        "__builtins__": __builtins__,
        "data": data_obj,
        "json": json,
        "re": re,
        "math": math,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "datetime": datetime,
        "date": date,
        "timedelta": timedelta,
    }


def extract_code_and_get_output(model_response: str, response_arr: Any) -> Any:
    try:
        start_idx = model_response.find("```python")
        if start_idx == -1:
            start_idx = model_response.find("def ")
            if start_idx == -1:
                raise ValueError("Python code block not found in response.")
            else:
                model_response_first_part = model_response
        else:
            model_response_first_part = model_response[start_idx + 9:]

        end_idx = model_response_first_part.find("```")
        if end_idx != -1:
            code = model_response_first_part[:end_idx].strip()
        else:
            code = model_response_first_part.strip()

        if "# Example usage:" in code:
            code = code.split("# Example usage:")[0]

        def_find = code.find("def")
        first_open_parenthesis = code.find("(")
        function_name = code[def_find + 4:first_open_parenthesis].strip()

        exec_globals = _build_exec_globals(response_arr)

        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                compiled = compile(ast.Module(body=[node], type_ignores=[]), filename="<ast>", mode="exec")
                exec(compiled, exec_globals)

        exec(code, exec_globals)

        if function_name not in exec_globals:
            raise ValueError(f"Function {function_name} not found after execution, code: {code}")

        func = exec_globals[function_name]
        try:
            return func(response_arr)
        except TypeError:
            # Function may take no args (uses 'data' global instead)
            return func()

    except Exception as e:
        logger.error(f"Error during code execution: {e}")
        return "Code execution error"
