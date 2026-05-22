"""Directed feedback for the self-correction loop.

Each error level (see :mod:`extensions.self_correction.error_detector`) gets a
specialised feedback message, following the slide design:

* L1 Hard Error      -- show the traceback, suggest ``.get()`` / key checks.
* L2 Empty Output    -- show a sample record, re-state the filter condition.
* L3 Format Mismatch -- state the required output format.
"""

from .error_detector import ErrorInfo, ErrorLevel


def build_feedback(code, error, schema=None, sample_record=None, question=None):
    """Return a feedback string appended to the prompt for the next round."""
    if error.level == ErrorLevel.HARD_ERROR:
        return _hard_error_feedback(code, error, schema, question)
    if error.level == ErrorLevel.EMPTY:
        return _empty_output_feedback(code, schema, sample_record, question)
    if error.level == ErrorLevel.FORMAT_MISMATCH:
        return _format_mismatch_feedback(error, question)
    return ""


def _hard_error_feedback(code, error, schema, question):
    return f"""Your previous script raised an error when it was executed.

Previous script:
```python
{code}
```

Error message:
```
{error.message}
```

Traceback:
```
{error.traceback or "N/A"}
```

JSON structure:
{(schema or "N/A")[:2000]}

Rewrite the script to fix this error. Common causes:
- accessing a key that may be absent -- use `.get()` or check membership first;
- a wrong nesting path into `data`;
- a wrong type (string vs number, list vs dict);
- a list index that is out of range.
Return only the corrected Python script; it must still print the answer."""


def _empty_output_feedback(code, schema, sample_record, question):
    return f"""Your previous script produced an empty result, but the JSON does
contain data. Re-check the traversal path and, above all, the filter condition.

Previous script:
```python
{code}
```

Question:
{question or "N/A"}

A sample record from the JSON:
{(sample_record or "N/A")[:1000]}

JSON structure:
{(schema or "N/A")[:1500]}

Re-read the filter condition stated in the question and make sure it matches the
real field names and values shown above (compare text case-insensitively).
Return only the corrected Python script."""


def _format_mismatch_feedback(error, question):
    return f"""Your previous script printed: {repr(error.raw_output)[:400]}

The question requires a different output format: {error.message}

Question:
{question or "N/A"}

Rewrite the script so the printed answer matches the required format -- for
example a plain integer for a count, a comma-separated list, or normalized text.
Return only the corrected Python script."""
