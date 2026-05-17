"""
Feedback Builder for Self-Debug

Constructs feedback prompts based on error type.
"""

from .error_detector import ErrorInfo, ErrorLevel


def build_feedback(code: str, error: ErrorInfo, schema: str = None,
                   sample_record: str = None, question: str = None) -> str:
    """
    Build feedback prompt for the model to fix its code.

    Args:
        code: The previous (buggy) code
        error: ErrorInfo from error detection
        schema: Optional schema for context
        sample_record: Optional sample data for context

    Returns:
        Feedback prompt string
    """
    if error.level == ErrorLevel.HARD_ERROR:
        return _build_hard_error_feedback(
            code, error.message, error.traceback, schema, sample_record, question
        )

    elif error.level == ErrorLevel.EMPTY:
        return _build_empty_output_feedback(code, schema, sample_record, question)

    elif error.level == ErrorLevel.FORMAT_MISMATCH:
        return _build_format_mismatch_feedback(code, error.raw_output, error.message, question)

    return ""


def _build_hard_error_feedback(code: str, error_message: str, traceback: str,
                               schema: str = None, sample_record: str = None,
                               question: str = None) -> str:
    """Build feedback for hard Python errors."""
    schema_snippet = schema[:2500] if schema else "N/A"
    sample = sample_record[:1500] if sample_record else "N/A"
    question_rules = _question_specific_rules(question)

    return f"""Your previous Python function had an error when executed.

Previous code:
```python
{code}
```

Question:
{question or "N/A"}

Error message:
```
{error_message}
```

Error traceback:
```
{traceback}
```

Relevant JSON structure:
{schema_snippet}

Sample data point:
{sample}

Please rewrite the function to fix this error. Pay attention to:
- Whether keys exist before accessing them (use .get() or try/except)
- Correct nesting structure of the JSON
- Data types (string vs number, list vs dict)
- List indices (ensure they're within bounds)
- If the root object is a dict, do not iterate over `data` directly unless the question really asks about top-level keys. Use the list path shown above, e.g. `data["available"]` or `data["data"]["result"]`.
- Use numeric `value` fields for calculations when available; formatted strings like "$12.30" are only display values.
- Do not invent filters or fields that are absent from the JSON. Match only conditions stated in the question.
- Normalize text comparisons on both sides with `.lower()`. Never compare an uppercased value with a lowercase literal.
- If this is a semantic validation error, follow the validator message exactly and rewrite the function. Do not return the old code.
- Return only the corrected Python code defining `solve(data)`.
{question_rules}

Write the corrected Python function:"""


def _build_empty_output_feedback(code: str, schema: str = None,
                                  sample_record: str = None,
                                  question: str = None) -> str:
    """Build feedback for empty/None output."""
    schema_snippet = schema[:2000] if schema else "N/A"
    sample = sample_record[:1000] if sample_record else "N/A"
    question_rules = _question_specific_rules(question)

    return f"""Your previous function returned an empty result, which seems incorrect for this query. The JSON contains relevant data — please review the structure and try a different traversal path.

Previous code:
```python
{code}
```

Question:
{question or "N/A"}

Schema (relevant excerpt):
{schema_snippet}

Sample data point:
{sample}

Correction rules:
- Use the actual list/dict path shown above; do not guess the root container.
- For booking room questions, match against `name` when present because it includes the full policy wording.
- For rates/prices, use numeric `value` fields when available instead of parsing display strings.
- Do not add constraints that are not asked in the question.
- Return only the corrected Python code defining `solve(data)`.
{question_rules}

Write a corrected Python function:"""


def _build_format_mismatch_feedback(code: str, raw_output: str,
                                     expected_format: str,
                                     question: str = None) -> str:
    """Build feedback for format mismatch."""
    return f"""Your previous function returned: {repr(raw_output)[:500]}

But the query asked for: {expected_format}

Question:
{question or "N/A"}

Please rewrite the function to return the answer in the requested format."""


def _question_specific_rules(question: str = None) -> str:
    return ""
