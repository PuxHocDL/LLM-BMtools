"""Score baseline predictions produced by `run_baseline.py`.

For every `*_predictions.json` under `experimental_scripts/results/predictions/`,
this script computes EM (`exact_match`) and Contains, then aggregates a summary
per (model, setup_type) pair.

LLM-as-a-judge requires a judge model. Pass `--judge "<model-id>"` to enable it.

    cd baseline
    python run_evaluation.py
    python run_evaluation.py --judge meta-llama/llama-3-3-70b-instruct
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experimental_scripts"))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

from experimental_scripts.qa_evaluation import calculate_exact_match, evaluate_sample  # noqa: E402
from generate_qa_pairs.tasks import evals  # noqa: E402
from generate_qa_pairs.tasks.utils import convert_dict_to_list_of_objects  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--predictions-dir",
                        default=os.path.join(ROOT, "experimental_scripts", "results", "predictions"))
    parser.add_argument("--evaluation-dir",
                        default=os.path.join(ROOT, "experimental_scripts", "results", "evaluation"))
    parser.add_argument("--judge", default=None,
                        help="Judge model id (omit to skip LLM-as-a-judge)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.evaluation_dir, exist_ok=True)
    wLLM = args.judge is not None

    summary = defaultdict(lambda: {"em": [], "contains": [], "judge": []})

    for filename in sorted(os.listdir(args.predictions_dir)):
        if not filename.endswith("_predictions.json"):
            continue
        pred_path = os.path.join(args.predictions_dir, filename)
        suffix = "eval.json" if wLLM else "eval_woLLM.json"
        eval_path = os.path.join(args.evaluation_dir, filename.replace("_predictions.json", "_") + suffix)

        try:
            with open(pred_path, "r") as f:
                raw = json.load(f)
            samples = convert_dict_to_list_of_objects(raw)
        except Exception as e:
            print(f"[skip] {filename} ({e})")
            continue

        for i, sample in enumerate(samples):
            raw[i]["metrics"]["exact_match"] = calculate_exact_match(sample)
            raw[i]["metrics"]["contains"] = evals.contains(sample)

        if wLLM:
            with Pool(processes=args.workers) as pool:
                results = pool.map(evaluate_sample, [(i, samples[i], args.judge) for i in range(len(samples))])
            for i, r in results:
                raw[i]["metrics"]["llm_as_a_judge"] = r

        with open(eval_path, "w") as f:
            json.dump(raw, f, indent=2)

        # Aggregate
        for row in raw:
            key = (row["model"], row["setup_type"])
            em = row["metrics"].get("exact_match")
            ctn = row["metrics"].get("contains")
            jdg = row["metrics"].get("llm_as_a_judge")
            if em is not None:
                summary[key]["em"].append(bool(em))
            if ctn is not None:
                summary[key]["contains"].append(bool(ctn))
            if jdg is not None:
                summary[key]["judge"].append(bool(jdg))

        print(f"[done] {filename}")

    print()
    print("=" * 80)
    print(f"{'Model':<45} {'Setup':<28} {'EM':>6} {'Cont':>6} {'Judge':>6}")
    print("=" * 80)
    for (model, setup), m in sorted(summary.items()):
        em = sum(m["em"]) / len(m["em"]) if m["em"] else float("nan")
        ctn = sum(m["contains"]) / len(m["contains"]) if m["contains"] else float("nan")
        jdg = sum(m["judge"]) / len(m["judge"]) if m["judge"] else float("nan")
        print(f"{model:<45} {setup:<28} {em:>6.3f} {ctn:>6.3f} {jdg:>6.3f}")


if __name__ == "__main__":
    main()
