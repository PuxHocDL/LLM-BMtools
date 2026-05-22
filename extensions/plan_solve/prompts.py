"""Plan-and-Solve prompt for code generation.

The model is asked to produce a single Python script structured as:

* ``PLAN``  -- ``#`` comments naming the relevant fields, the filter
  conditions and the aggregation needed to answer the question.
* ``SOLVE`` -- code that implements each planned step, storing intermediate
  results in clearly named variables.

The script ends with ``print(<answer>)``; the printed value is the answer.
Everything happens in a single LLM call, after which the script is executed
(see :mod:`extensions.code_exec`).
"""

PLAN_SOLVE_TEMPLATE = '''You answer a question about a JSON API response by writing one Python script.

The JSON object is already loaded into a variable named `data`.
Write the script in two clearly separated parts:

# PLAN
# Use `#` comments to state, in order:
#   - which field(s)/path(s) inside `data` hold the information,
#   - the filter condition(s), if any,
#   - the aggregation or computation needed (count, sum, min, max, average, list, ...).

# SOLVE
Then write the code that carries out each planned step. Store intermediate
results in clearly named variables, and at the end print the answer with `print()`.

Rules for the printed answer:
- Print ONLY the answer value, nothing else.
- Numbers: a plain number, no units, no thousands separators.
- Names / dates / strings: the value verbatim, original casing and diacritics, no quotes.
- Lists: comma-separated with no space after the comma, in JSON order; keep
  duplicates unless the question asks for unique values.
- If nothing matches: print `0` for counts, otherwise print `none`.

Return ONLY the Python script inside one ```python code block.

JSON DATA:
```json
{json_str}
```

QUESTION:
{question}
'''


def build_plan_solve_prompt(json_str, question):
    """Build the single-call Plan-and-Solve code-generation prompt."""
    return PLAN_SOLVE_TEMPLATE.format(json_str=json_str, question=question)
