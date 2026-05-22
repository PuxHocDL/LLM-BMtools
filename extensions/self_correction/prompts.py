"""Initial code-generation prompt for the self-correction loop.

The model writes a Python script that reads the global ``data`` object and
prints the answer. If the script fails, errors, or returns nothing, the debug
loop appends directed feedback (see :mod:`feedback_builder`) and asks again.
"""

CODE_TEMPLATE = '''You answer a question about a JSON API response by writing one Python script.

The JSON object is already loaded into a variable named `data`.
Write a self-contained script that extracts the answer from `data` and prints
it with `print()`.

Rules:
- Be defensive on missing keys: use `.get()` or check membership before access.
- Search every record, not just the first one.
- Compare text case-insensitively unless the question asks for an exact id.
- Print ONLY the answer value, nothing else.
- Numbers: a plain number, no units, no thousands separators.
- Names / dates / strings: the value verbatim, original casing and diacritics.
- Lists: comma-separated with no space after the comma, in JSON order.
- If nothing matches: print `0` for counts, otherwise print `none`.

Return ONLY the Python script inside one ```python code block.

JSON DATA:
```json
{json_str}
```

QUESTION:
{question}
'''


def build_code_prompt(json_str, question):
    """Build the round-0 code-generation prompt for the self-correction loop."""
    return CODE_TEMPLATE.format(json_str=json_str, question=question)
