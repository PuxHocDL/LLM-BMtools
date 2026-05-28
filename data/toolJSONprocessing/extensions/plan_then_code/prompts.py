"""Prompts for Plan-then-Code (Decomposition / Least-to-Most) on pruned JSON."""

PLAN_SYSTEM = (
    "You are a planning assistant for a JSON data extraction task. "
    "You decompose a user question into a short, ordered list of atomic operations "
    "(locate / filter / aggregate / project / format) that can be executed against the JSON."
)

PLAN_USER_TEMPLATE = """You are given a JSON SCHEMA (the shape of the data) and a QUESTION about the JSON.

Output a numbered plan of 2 to 5 steps to answer the question.
Rules:
- Each step is one atomic operation: locate-path, filter-by-condition, count, sum, max, min, average, sort, project-field, or format.
- Reference JSON paths from the schema when relevant (e.g. `$.products[*].price`).
- Be specific about conditions ("where rating > 4", not "where rating is high").
- Do NOT solve the question. Do NOT output the answer. Plan only.
- Keep each step on a single line, prefixed with its number and a period.

JSON SCHEMA:
{schema}

QUESTION:
{question}

PLAN:
"""

PLAN_AND_SOLVE_TEMPLATE = """You are given a JSON data object (already pruned to relevant content) and a QUESTION.
Solve it using PLAN-AND-SOLVE prompting:
  1) PLAN: write a short numbered plan (2 to 5 atomic steps) to answer the question from the JSON.
  2) SOLVE: execute each step over the JSON, briefly noting intermediate findings.
  3) On the LAST line write exactly: FINAL ANSWER: <value>

Output format rules:
- For numbers, just the number (no units, no thousands separators).
- For names/dates/strings, just the value verbatim, in original casing and diacritics (e.g. `Résidence`, not `R\\u00e9sidence`). No surrounding quotes.
- For lists, comma-separated with NO space after the comma (e.g. `A,B,C`). Preserve JSON order. Keep duplicates UNLESS the question asks for unique values.
- The FINAL ANSWER line MUST contain a value. Never write `FINAL ANSWER:` alone. If nothing matches, write `FINAL ANSWER: 0` for counts, `FINAL ANSWER: none` otherwise.
- Nothing after the FINAL ANSWER line.

--- WORKED EXAMPLES ---
Example 1 (extractive, scalar):
JSON DATA: {{"results":[{{"id":"x1","price":42.5,"city":"Hanoi"}}]}}
QUESTION: What is the price of x1?
PLAN:
1. Locate the record with id == "x1".
2. Read its "price" field.
SOLVE:
1. Found x1 in results[0].
2. price = 42.5.
FINAL ANSWER: 42.5

Example 2 (filtering, list):
JSON DATA: {{"items":[{{"id":"a","tag":"x"}},{{"id":"b","tag":"y"}},{{"id":"c","tag":"x"}}]}}
QUESTION: List item ids where tag is "x". Output a comma separated list.
PLAN:
1. Iterate items.
2. Keep id when tag == "x".
3. Join with commas, preserve order.
SOLVE:
1. items has 3 entries.
2. Matches: a, c.
3. Joined: a,c.
FINAL ANSWER: a,c

Example 3 (aggregation, count):
JSON DATA: {{"rooms":[{{"avail":true}},{{"avail":false}},{{"avail":true}}]}}
QUESTION: How many rooms are available?
PLAN:
1. Count rooms with avail == true.
SOLVE:
1. rooms[0] true, rooms[1] false, rooms[2] true -> count = 2.
FINAL ANSWER: 2
--- END EXAMPLES ---
{candidate_block}
JSON DATA:
{json_str}

QUESTION:
{question}

PLAN:
1."""


def build_answer_prompt(plan: str, json_str: str, question: str, summary_answer=None) -> str:
    """Single-stage Plan-and-Solve prompt. `plan` arg kept for API compatibility but ignored
    (the LLM produces its own plan inline). Avoids the 2-call planner+executor pattern which
    overloaded the gpt-oss-20b endpoint and confused the model with conflicting instructions.
    """
    candidate_block = ""
    if summary_answer is not None:
        candidate_block = (
            f"\nDeterministic candidate from pruner: {summary_answer}\n"
            "If your plan's execution agrees with this candidate, output exactly the candidate.\n"
        )
    return PLAN_AND_SOLVE_TEMPLATE.format(
        json_str=json_str,
        question=question,
        candidate_block=candidate_block,
    )


def extract_final_answer(response: str) -> str:
    """Pull `FINAL ANSWER: <x>` from response. Falls back to last non-empty line."""
    if not response:
        return ""
    marker_idx = response.rfind("FINAL ANSWER:")
    if marker_idx >= 0:
        tail = response[marker_idx + len("FINAL ANSWER:"):].strip()
        if tail:
            return tail.splitlines()[0].strip().strip("`").strip()
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    return lines[-1] if lines else response.strip()
