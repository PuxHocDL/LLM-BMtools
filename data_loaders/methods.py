"""Data loaders for the three improvement methods on the ToolJSON benchmark.

Each loader builds on :class:`ToolJSONLoader` and applies HeuristicPlus JSON
pruning to the raw API response first:

* :class:`PruningLoader`        -- prune, then answer directly.
* :class:`PlanSolveLoader`      -- prune, generate a PLAN+SOLVE script, execute it.
* :class:`SelfCorrectionLoader` -- prune, generate code, run the debug loop.
"""

import json
import os
import threading

from data_loaders.tooljson import ToolJSONLoader
from extensions.code_exec import extract_code, run_code
from extensions.plan_solve import build_plan_solve_prompt
from extensions.pruning import heuristic_plus_prune


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_raw_json(sample, base_repo_dir):
    """Read the raw API-response JSON file referenced by a ToolJSON sample."""
    api_response_path = sample.get("api_response_path", "")
    path = os.path.join(
        base_repo_dir, "generate_qa_pairs", "data", "api_responses",
        os.path.basename(api_response_path),
    )
    path = path.replace("?", "_").replace(":", "_").replace('"', "_")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _json_shape(obj, path="data", depth=0, lines=None):
    """Compact structural summary of a JSON object (paths + types)."""
    if lines is None:
        lines = []
    if len(lines) >= 60 or depth > 5:
        return lines
    if isinstance(obj, dict):
        lines.append(f"{path}: object keys={list(obj.keys())[:15]}")
        for key, value in list(obj.items())[:12]:
            _json_shape(value, f"{path}.{key}", depth + 1, lines)
    elif isinstance(obj, list):
        lines.append(f"{path}: array len={len(obj)}")
        if obj:
            _json_shape(obj[0], f"{path}[0]", depth + 1, lines)
    else:
        lines.append(f"{path}: {type(obj).__name__} = {repr(obj)[:60]}")
    return lines


def _answer_prompt(question, pruned_json_str, summary_answer=None):
    """Prompt for the pruning-only method: read the JSON, output the value."""
    prompt = (
        "Answer the question using the JSON data below.\n"
        "Output only the answer value -- a number, a name/date, or a "
        "comma-separated list. Do not explain.\n\n"
        f"JSON data:\n{pruned_json_str}\n\n"
        f"Question: {question}\n"
    )
    if summary_answer is not None:
        prompt += (
            f"\nA deterministic pre-computed candidate answer is: {summary_answer}\n"
            "If it is consistent with the JSON, return exactly that value.\n"
        )
    return prompt + "Answer:"


# ---------------------------------------------------------------------------
# Base loader: load + HeuristicPlus pruning
# ---------------------------------------------------------------------------
class _PrunedLoader(ToolJSONLoader):
    """Shared base: prunes each sample's API response with HeuristicPlus."""

    stop_sequences = []
    enable_thinking = False
    max_tokens = 1024

    def __init__(self, data_path, base_repo_dir="data/toolJSONprocessing",
                 agent_name=None, budget=10000):
        super().__init__(data_path, base_repo_dir, agent_name=agent_name)
        self.budget = budget
        self.total_original_chars = 0
        self.total_pruned_chars = 0
        self.samples_processed = 0
        self._stats_lock = threading.Lock()

    def _prune(self, sample):
        """Return ``(pruned_obj, pruned_json_str)`` for a sample."""
        question = sample.get("question", "")
        raw = _load_raw_json(sample, self.base_repo_dir)
        json_obj = json.loads(raw)
        pruned = heuristic_plus_prune(
            json_obj, question,
            max_chars=self.budget,
            query_context=sample.get("api_query"),
        )
        pruned_str = json.dumps(pruned, indent=2, ensure_ascii=False)
        with self._stats_lock:
            self.total_original_chars += len(raw)
            self.total_pruned_chars += len(pruned_str)
            self.samples_processed += 1
        return pruned, pruned_str


# ---------------------------------------------------------------------------
# Method (I): JSON Pruning
# ---------------------------------------------------------------------------
class PruningLoader(_PrunedLoader):
    """HeuristicPlus JSON pruning, then a direct answer."""

    def format_prompt(self, sample):
        try:
            pruned, pruned_str = self._prune(sample)
        except Exception:
            return ToolJSONLoader.format_prompt(self, sample)
        summary = None
        if isinstance(pruned, dict):
            summary = pruned.get("__question_summary__", {}).get("answer")
        prompt = _answer_prompt(sample.get("question", ""), pruned_str, summary)
        return [{"role": "user", "content": prompt}]


# ---------------------------------------------------------------------------
# Method (II): Plan-and-Solve with code generation
# ---------------------------------------------------------------------------
class PlanSolveLoader(_PrunedLoader):
    """Pruning + Plan-and-Solve: generate one PLAN+SOLVE script and execute it."""

    max_tokens = 2048

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # format_prompt and postprocess_prediction run in the same worker
        # thread for a given sample, so the pruned payload is handed over here.
        self._tls = threading.local()

    def format_prompt(self, sample):
        try:
            pruned, pruned_str = self._prune(sample)
        except Exception:
            self._tls.data = None
            return ToolJSONLoader.format_prompt(self, sample)
        self._tls.data = pruned
        prompt = build_plan_solve_prompt(pruned_str, sample.get("question", ""))
        return [{"role": "user", "content": prompt}]

    def postprocess_prediction(self, prediction):
        data = getattr(self._tls, "data", None)
        if data is None:
            return ""
        code = extract_code(str(prediction or ""))
        try:
            return run_code(code, data)
        except Exception:
            # No retry loop for Plan-and-Solve: a failed script is a wrong answer.
            return ""


# ---------------------------------------------------------------------------
# Method (III): Self-Correction with execution feedback
# ---------------------------------------------------------------------------
class SelfCorrectionLoader(_PrunedLoader):
    """Pruning + Self-Correction: generate code, then run the debug loop.

    The evaluator detects :meth:`execute_debug_loop` and routes generation
    through it instead of a single LLM call.
    """

    max_tokens = 2048
    max_rounds = 3

    def format_prompt(self, sample):
        # Unused -- generation is driven by execute_debug_loop. Kept trivial
        # so the evaluator's mandatory format_prompt call stays cheap.
        return [{"role": "user", "content": ""}]

    def execute_debug_loop(self, client, sample_id, sample):
        from extensions.self_correction import build_code_prompt, debug_loop

        question = sample.get("question", "")
        try:
            pruned, pruned_str = self._prune(sample)
        except Exception:
            return ""

        prompt_orig = build_code_prompt(pruned_str, question)
        schema = "\n".join(_json_shape(pruned))

        def model_generate(prompt, temperature=0.3, **_kwargs):
            response = client.generate(
                prompt, stop=[], temperature=temperature,
                max_tokens=self.max_tokens, max_retries=2,
            )
            return extract_code(response)

        def execute_code(code, data):
            return run_code(code, data)

        final_output, _history = debug_loop(
            model_generate=model_generate,
            execute_code=execute_code,
            prompt_orig=prompt_orig,
            api_response=pruned,
            question=question,
            schema=schema,
            max_rounds=self.max_rounds,
            error_levels=(1, 2, 3),
            temperature=0.3,
        )
        if final_output is None:
            return ""
        return str(final_output).strip()
