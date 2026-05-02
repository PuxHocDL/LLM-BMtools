"""
Generate RESULTS.md from all result JSON files in results/.
"""
import json
import os
from datetime import datetime

RESULTS_DIR = "results"
METRICS = ["exact_match", "contains", "f1_score", "action_match", "llm_judge"]
METRIC_SHORT = {
    "exact_match": "EM",
    "contains": "Contains",
    "f1_score": "F1",
    "action_match": "Act Match",
    "llm_judge": "LLM Judge",
}

KNOWN_DATASETS = ["ToolBench", "ComplexFunc", "ToolJSON"]


def classify_file(fname):
    base = fname.replace("_detailed.json", "")
    model = "unknown"
    prefix = base
    for ds in KNOWN_DATASETS:
        idx = base.find(ds)
        if idx != -1:
            after_ds = base[idx + len(ds):]
            after_ds = after_ds.replace("Loader", "")
            if after_ds.startswith("_"):
                model = after_ds[1:]
            prefix = base[:idx + len(ds)]
            prefix = prefix.replace("Loader", "")
            break

    if prefix.startswith("Enhanced"):
        strategy = "Enhanced"
        dataset = prefix.replace("Enhanced", "")
    elif any(prefix == ds for ds in KNOWN_DATASETS):
        strategy = "Baseline"
        dataset = prefix
    else:
        parts = prefix.split("_", 1)
        strategy = parts[0]
        dataset = parts[1] if len(parts) > 1 else prefix

    return strategy, dataset, model


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"No {RESULTS_DIR}/ directory found.")
        return

    files = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith("_detailed.json"))
    if not files:
        print("No result files found.")
        return

    rows = []
    for fname in files:
        path = os.path.join(RESULTS_DIR, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        strategy, dataset, model = classify_file(fname)
        n = len(data.get("details", []))
        rows.append({
            "file": fname,
            "strategy": strategy,
            "dataset": dataset,
            "model": model,
            "n": n,
            "summary": data.get("summary", {}),
        })

    models = sorted(set(r["model"] for r in rows))

    lines = []
    lines.append("# Evaluation Results\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    for mdl in models:
        mdl_rows = [r for r in rows if r["model"] == mdl]
        lines.append(f"## Model: `{mdl}`\n")

        datasets = sorted(set(r["dataset"] for r in mdl_rows))

        for ds in datasets:
            ds_rows = [r for r in mdl_rows if r["dataset"] == ds]
            ds_rows.sort(key=lambda r: r["strategy"])

            lines.append(f"### {ds}\n")

            header = "| Strategy | Samples | " + " | ".join(METRIC_SHORT[m] for m in METRICS) + " |"
            separator = "|" + "|".join(["---"] * (2 + len(METRICS))) + "|"
            lines.append(header)
            lines.append(separator)

            for r in ds_rows:
                vals = " | ".join(f"{r['summary'].get(m, 0):.4f}" for m in METRICS)
                lines.append(f"| {r['strategy']} | {r['n']} | {vals} |")

            lines.append("")

    md_content = "\n".join(lines)
    with open("RESULTS.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"RESULTS.md generated with {len(rows)} entries.")


if __name__ == "__main__":
    main()
