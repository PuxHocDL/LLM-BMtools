"""Evaluation pipeline for enhancing LLM processing of JSON tool outputs.

Runs one method over the ToolJSON benchmark and reports EM / Contains / F1
/ LLM-as-a-judge.

    python main.py --method plan_solve --model granite --judge llama

Methods:
    baseline         -- raw API JSON, answer directly (no pruning).
    pruning          -- HeuristicPlus JSON pruning, then answer.
    plan_solve       -- pruning + Plan-and-Solve code generation.
    self_correction  -- pruning + code generation with execution feedback.
"""

import argparse

from data_loaders.tooljson import ToolJSONLoader
from data_loaders.methods import PlanSolveLoader, PruningLoader, SelfCorrectionLoader
from evaluators.evaluator import Evaluator

DATA_PATH = "data/toolJSONprocessing/generate_qa_pairs/data/qa_pairs"

METHODS = {
    "baseline": ToolJSONLoader,
    "pruning": PruningLoader,
    "plan_solve": PlanSolveLoader,
    "self_correction": SelfCorrectionLoader,
}

METHOD_LABELS = {
    "baseline": "Baseline (raw JSON, direct answer)",
    "pruning": "HeuristicPlus JSON Pruning",
    "plan_solve": "JSON Pruning + Plan-and-Solve (code generation)",
    "self_correction": "JSON Pruning + Self-Correction (execution feedback)",
}


def main():
    parser = argparse.ArgumentParser(
        description="Enhancing LLM processing of JSON tool outputs."
    )
    parser.add_argument("--method", choices=list(METHODS), default="baseline",
                        help="Improvement method to evaluate")
    parser.add_argument("--model", default="granite",
                        help="Agent model name from config.yaml")
    parser.add_argument("--judge", default="llama",
                        help="Judge model name from config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N samples (dry-run)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Concurrent worker threads")
    parser.add_argument("--budget", type=int, default=10000,
                        help="HeuristicPlus pruning character budget")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip the LLM-as-a-judge metric")
    args = parser.parse_args()

    print("=" * 60)
    print(f"METHOD : {METHOD_LABELS[args.method]}")
    print(f"MODEL  : {args.model}")
    print("=" * 60)

    loader_cls = METHODS[args.method]
    if args.method == "baseline":
        loader = loader_cls(DATA_PATH, agent_name=args.model)
    else:
        loader = loader_cls(DATA_PATH, agent_name=args.model, budget=args.budget)

    evaluator = Evaluator(data_loader=loader, agent_name=args.model, judge_name=args.judge)
    evaluator.evaluate(limit=args.limit, workers=args.workers, skip_judge=args.skip_judge)

    if getattr(loader, "samples_processed", 0) > 0:
        avg_orig = loader.total_original_chars / loader.samples_processed
        avg_pruned = loader.total_pruned_chars / loader.samples_processed
        print("\n--- JSON Pruning ---")
        print(f"Avg original size : {avg_orig:,.0f} chars")
        print(f"Avg pruned size   : {avg_pruned:,.0f} chars "
              f"({avg_pruned / avg_orig * 100:.2f}% of original)")


if __name__ == "__main__":
    main()
