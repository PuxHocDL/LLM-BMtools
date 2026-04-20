"""
Rescore all result files with the improved exact_match metric,
update the JSON/CSV files, and print a comparison table.
"""

import json
import os
import csv
from core.metrics import Metrics

RESULTS_DIR = "results"
METRICS = ["exact_match", "contains", "f1_score", "action_match", "llm_judge"]


def rescore_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    details = data["details"]
    old_summary = dict(data["summary"])

    # Recompute per-sample metrics
    for s in details:
        pred = s.get("prediction", "")
        gt = s.get("ground_truth", "")
        s["exact_match"] = Metrics.exact_match(pred, gt)
        s["contains"] = Metrics.contains(pred, gt)
        s["f1_score"] = Metrics.f1_score(pred, gt)
        s["action_match"] = Metrics.action_match(pred, gt)
        # llm_judge is not recomputed (requires API call)

    # Recompute summary
    n = len(details)
    new_summary = {}
    for key in data["summary"]:
        vals = [s.get(key, 0) for s in details]
        new_summary[key] = sum(vals) / n if n else 0
    data["summary"] = new_summary

    # Save updated JSON
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Save updated CSV
    csv_path = path.replace("_detailed.json", "_detailed.csv")
    if details:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=details[0].keys())
            writer.writeheader()
            writer.writerows(details)

    return old_summary, new_summary, n


def classify_file(fname):
    """Return (strategy, dataset, model) from filename like S1_ToolBench_qwen_detailed.json."""
    # Strip suffix
    base = fname.replace("_detailed.json", "")  # e.g. "S1_ToolBench_qwen"

    # Extract model name (last segment after the loader/strategy part)
    # Known loader suffixes that end the "class name" part
    known_datasets = ["ToolBench", "ComplexFunc", "ToolJSON"]

    model = "unknown"
    prefix = base
    for ds in known_datasets:
        # Find dataset name in the base, model is everything after it + "_"
        idx = base.find(ds)
        if idx != -1:
            after_ds = base[idx + len(ds):]
            # Handle "Loader_model" or just "_model"
            after_ds = after_ds.replace("Loader", "")
            if after_ds.startswith("_"):
                model = after_ds[1:]
            prefix = base[:idx + len(ds)]
            # Remove trailing "Loader" from prefix
            prefix = prefix.replace("Loader", "")
            break

    # Classify strategy
    if prefix.startswith("Enhanced"):
        strategy = "Enhanced"
        dataset = prefix.replace("Enhanced", "")
    elif any(prefix == ds for ds in known_datasets):
        strategy = "Baseline"
        dataset = prefix
    else:
        # S1_ToolBench, S2_ComplexFunc, etc.
        parts = prefix.split("_", 1)
        strategy = parts[0]
        dataset = parts[1] if len(parts) > 1 else prefix

    return strategy, dataset, model


def main():
    files = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith("_detailed.json"))

    # Collect all results
    rows = []
    for fname in files:
        path = os.path.join(RESULTS_DIR, fname)
        old_sum, new_sum, n = rescore_file(path)
        strategy, dataset, model = classify_file(fname)
        rows.append({
            "file": fname,
            "strategy": strategy,
            "dataset": dataset,
            "model": model,
            "n": n,
            "old": old_sum,
            "new": new_sum,
        })

    # Group by model
    models = sorted(set(r["model"] for r in rows))

    for mdl in models:
        mdl_rows = [r for r in rows if r["model"] == mdl]
        print(f"\n{'#'*80}")
        print(f"  MODEL: {mdl}")
        print(f"{'#'*80}")

        # Print compact summary table per model
        datasets = sorted(set(r["dataset"] for r in mdl_rows))
        header = f"{'Strategy':<12} {'Dataset':<12}"
        for m in METRICS:
            short = m.replace("exact_match", "EM").replace("contains", "Cont") \
                     .replace("f1_score", "F1").replace("action_match", "ActM") \
                     .replace("llm_judge", "Judge")
            header += f" {short:>7}"
        print(header)
        print("-" * len(header))

        for ds in datasets:
            ds_rows = [r for r in mdl_rows if r["dataset"] == ds]
            ds_rows.sort(key=lambda r: r["strategy"])
            for r in ds_rows:
                line = f"{r['strategy']:<12} {r['dataset']:<12}"
                for m in METRICS:
                    v = r["new"].get(m, 0)
                    line += f" {v:>7.3f}"
                print(line)
            print()

    print(f"Files updated in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
