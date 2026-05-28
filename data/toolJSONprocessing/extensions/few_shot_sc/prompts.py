def get_few_shot_prompt(schema_str, json_str, query, exemplars):
    prompt = (
        "You will be given a JSON object as data which is a response from a REST API.\n"
        "Your task is to extract and return information from the JSON object that\n"
        "answers the user query.\n\n"
    )
    if exemplars:
        prompt += f"Below are {len(exemplars)} examples of similar tasks:\n\n"
        for i, ex in enumerate(exemplars, 1):
            prompt += f"[EXAMPLE {i}]\n"
            prompt += f"JSON schema: {ex.schema}\n"
            prompt += f"Query: {ex.question}\n"
            prompt += f"Python Function:\n```python\n{ex.code}\n```\n\n"

    prompt += "Now, write a Python function for the following:\n\n"
    prompt += f"JSON schema: {schema_str}\n"
    prompt += f"data = {json_str}\n"
    prompt += f"Query: {query}\n\n"
    prompt += "Python Function:\n```python\n"
    return prompt


def get_few_shot_direct_prompt(schema_str, json_str, query, exemplars):
    """Direct-answer style: ask model to output the value directly, no code."""
    prompt = (
        "You will analyze a JSON output from a REST API to answer the user's question.\n"
        "Output ONLY the final answer value (a name, number, date, or comma-separated list). "
        "No explanation, no code, no JSON.\n\n"
    )
    if exemplars:
        prompt += f"Below are {len(exemplars)} examples:\n\n"
        for i, ex in enumerate(exemplars, 1):
            prompt += f"[EXAMPLE {i}]\n"
            prompt += f"Query: {ex.question}\n"
            prompt += f"Answer: {ex.answer}\n\n"

    prompt += "Now answer this:\n\n"
    prompt += f"JSON Output:\n{json_str}\n\n"
    prompt += f"Question: {query}\n"
    prompt += "Answer:"
    return prompt


def get_few_shot_direct_cot_prompt(schema_str, json_str, query, exemplars):
    """Chain-of-Thought direct-answer style: reasoning then final answer."""
    prompt = (
        "You will analyze a JSON output from a REST API to answer the user's question.\n"
        "First, think step by step:\n"
        "  1. Identify the relevant fields in the JSON for the question.\n"
        "  2. Determine the operation needed (lookup / filter / count / sum / max / list).\n"
        "  3. Perform the operation mentally over the JSON.\n"
        "Then on the LAST line, output exactly:\n"
        "FINAL ANSWER: <value>\n"
        "where <value> is a single name, number, date, or comma-separated list.\n"
        "No extra text after FINAL ANSWER.\n\n"
    )
    if exemplars:
        prompt += f"Below are {len(exemplars)} examples:\n\n"
        for i, ex in enumerate(exemplars, 1):
            prompt += f"[EXAMPLE {i}]\n"
            prompt += f"Query: {ex.question}\n"
            prompt += f"FINAL ANSWER: {ex.answer}\n\n"

    prompt += "Now answer this:\n\n"
    prompt += f"JSON Output:\n{json_str}\n\n"
    prompt += f"Question: {query}\n"
    prompt += "Reasoning:"
    return prompt


def get_verify_cot_prompt(schema_str, json_obj, query, exemplars):
    """A3-style heuristic-candidate prompt with optional few-shot exemplars prepended.
    Asks model to return the candidate verbatim — minimal second-guessing.
    """
    import json as _json

    candidate = None
    if isinstance(json_obj, dict):
        summary = json_obj.get("__question_summary__")
        if isinstance(summary, dict):
            candidate = summary.get("answer")

    try:
        json_str = _json.dumps(json_obj, indent=2, default=str, ensure_ascii=False)
    except Exception:
        json_str = str(json_obj)

    prompt = "Analyze the following JSON output to answer the user's question.\n"
    prompt += "If the JSON contains __question_summary__.answer, return exactly that value and nothing else.\n"
    prompt += "Otherwise, answer with only the requested value, count, or comma-separated list. Do not explain.\n"

    if exemplars:
        prompt += "\nExamples of correctly answered queries:\n"
        for i, ex in enumerate(exemplars, 1):
            prompt += f"  Q: {ex.question}\n"
            prompt += f"  A: {ex.answer}\n"

    prompt += f"JSON Output:\n{json_str}\n\n"
    prompt += f"Question:\n{query}\n"
    if candidate is not None:
        prompt += f"Extracted answer candidate:\n{candidate}\n"
        prompt += "Return exactly the extracted answer candidate and nothing else."
    else:
        prompt += "Answer directly based on the JSON content."
    return prompt


def extract_final_answer(response: str) -> str:
    """Pull 'FINAL ANSWER: <x>' from CoT response. Falls back to last non-empty line."""
    if not response:
        return ""
    marker = "FINAL ANSWER:"
    idx = response.rfind(marker)
    if idx >= 0:
        tail = response[idx + len(marker):].strip()
        # Take first line only — model sometimes adds explanation after
        return tail.splitlines()[0].strip() if tail else ""
    # Fallback: last non-empty line
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    return lines[-1] if lines else response.strip()
