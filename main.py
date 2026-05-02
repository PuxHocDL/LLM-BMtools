import argparse
from data_loaders.toolbench import ToolBenchLoader
from data_loaders.complexfunc import ComplexFuncLoader
from data_loaders.tooljson import ToolJSONLoader
from evaluators.evaluator import Evaluator
import yaml

DATA_PATHS = {
    "toolbench": "data/ToolBench-Static/in_domain.jsonl",
    "complexfunc": "data/ComplexFuncBench.jsonl",
    "tooljson": "data/toolJSONprocessing/generate_qa_pairs/data/qa_pairs",
}

BASELINE_LOADERS = {
    "toolbench": ToolBenchLoader,
    "complexfunc": ComplexFuncLoader,
    "tooljson": ToolJSONLoader,
}

def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Pipeline")
    parser.add_argument("--dataset", type=str, required=True, choices=["toolbench", "complexfunc", "tooljson", "all"], help="Dataset to evaluate on")
    parser.add_argument("--model", type=str, default="qwen", help="Agent model name from config.yaml")
    parser.add_argument("--judge", type=str, default=None, help="Judge model name from config.yaml (defaults to --model)")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to evaluate (for dry-run)")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers for multi-threading")
    parser.add_argument("--enhanced", action="store_true", help="Enable Enhanced Mode (JSONPruning and Semantic Retrieval)")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["s1_prompt", "s2_compress", "s3_cot", "s4_twostage", "s5_fewshot", "s6_context"],
                        help="Enhancement strategy to use (overrides --enhanced)")
    args = parser.parse_args()
    if args.judge is None:
        args.judge = args.model

    datasets_to_run = ["toolbench", "complexfunc", "tooljson"] if args.dataset == "all" else [args.dataset]

    if args.strategy:
        from data_loaders.strategies import STRATEGIES
        strat = STRATEGIES[args.strategy]
        print("\n" + "="*50)
        print(f"STRATEGY: {strat['label']}")
        print("="*50)
    elif args.enhanced:
        print("\n" + "="*50)
        print("🚀 RUNNING IN ENHANCED MODE (PAPER MODE) 🚀")
        print("="*50)

    for ds in datasets_to_run:
        print(f"\n{'='*50}\nEVALUATING DATASET: {ds.upper()}\n{'='*50}")
        data_path = DATA_PATHS[ds]

        if args.strategy:
            from data_loaders.strategies import STRATEGIES
            strat = STRATEGIES[args.strategy]
            loader_cls = strat[ds]
            loader = loader_cls(data_path, agent_name=args.model)
        elif args.enhanced:
            if ds == "toolbench":
                from data_loaders.enhanced_loaders import EnhancedToolBenchLoader
                loader = EnhancedToolBenchLoader(data_path, agent_name=args.model)
            elif ds == "tooljson":
                from data_loaders.enhanced_loaders import EnhancedToolJSONLoader
                loader = EnhancedToolJSONLoader(data_path, agent_name=args.model)
            else:
                loader = BASELINE_LOADERS[ds](data_path, agent_name=args.model)
        else:
            loader = BASELINE_LOADERS[ds](data_path, agent_name=args.model)

        evaluator = Evaluator(data_loader=loader, agent_name=args.model, judge_name=args.judge)
        evaluator.evaluate(limit=args.limit, workers=args.workers)

if __name__ == "__main__":
    main()
