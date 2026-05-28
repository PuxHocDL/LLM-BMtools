"""Stage-1 planner: question + schema -> ordered atomic steps."""

import re
from .prompts import PLAN_SYSTEM, PLAN_USER_TEMPLATE


_STEP_LINE_RE = re.compile(r"^\s*\d+[.)]\s+")


def _clean_plan(raw: str) -> str:
    """Keep only numbered-step lines; cap at 6 steps to bound latency."""
    if not raw:
        return ""
    lines = []
    for line in raw.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        if _STEP_LINE_RE.match(line):
            lines.append(line.strip())
            if len(lines) >= 6:
                break
    if not lines:
        return raw.strip()
    return "\n".join(lines)


def generate_plan(question: str, schema_str: str, llm_client, max_tokens: int = 400) -> str:
    """Call LLM to produce a numbered plan. Returns cleaned plan text."""
    user_prompt = PLAN_USER_TEMPLATE.format(schema=schema_str, question=question)
    messages = [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    raw = llm_client.generate(
        messages,
        stop=["\n\nJSON", "\nQUESTION:", "\nPLAN:"],
        max_tokens=max_tokens,
    )
    if isinstance(raw, str) and raw.startswith("__ERROR"):
        return ""
    return _clean_plan(raw or "")
