"""Thin entry point to re-run the paper baseline (Kate et al., 2025).

This wraps `experimental_scripts/qa_inference.py` and runs the three baseline
setups reported in our report:

    code            -> setup_type = "code_generation"
    code_schema     -> setup_type = "code_generation_schema"
    answer          -> setup_type = "direct_prompting"

After inference, run `python run_evaluation.py` to score predictions
(EM / Contains / optional LLM-as-a-judge).

Usage:

    cd baseline
    python run_baseline.py --setup code_schema --model openai/gpt-oss-20b
    python run_baseline.py --setup answer      --model ibm-granite/granite-3.3-8b-instruct
    python run_baseline.py --setup code        --model mistralai/Devstral-Small-2507

The script expects environment variables from `.env` (copy from `example.env`)
and the ToolJSON benchmark under
`../data/toolJSONprocessing/generate_qa_pairs/data/`.
"""

import argparse
import importlib
import json
import os
import sys
import types
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experimental_scripts"))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

from experimental_scripts.qa_inference import run_inference, get_api_response_cf  # noqa: E402
from generate_qa_pairs.tasks.data_structures import LongResponseQASample  # noqa: E402


SETUP_ALIASES = {
    "code": "code_generation",
    "code_schema": "code_generation_schema",
    "answer": "direct_prompting",
    "answer_schema": "direct_prompting_schema",
}

TASK_LISTS = [
    "booking-com15.p.rapidapi.com_Search_Hotels_By_Coordinates",
    "booking-com15.p.rapidapi.com_Search_Car_Rentals",
    "booking-com15.p.rapidapi.com_Get_Seat_Map",
    "real-time-product-search.p.rapidapi.com_search?",
    "last10k-company-v1.p.rapidapi.com_v1_company_filings",
    "booking-com15.p.rapidapi.com_Get_Room_List_With_Availability",
]

DEFAULT_LLM_PARAMETERS = {
    "max_new_tokens": 1000,
    "min_new_tokens": 1,
    "top_p": 0.1,
    "temperature": 0.0,
    "random_seed": 1,
    "decoding_method": "greedy",
    "stop_sequences": [],
}


def build_qa_pairs(task_file: str, data_root: str, simplify_json: bool, task: str):
    with open(task_file, "r") as f:
        qa_pairs = json.load(f)

    out = []
    for sample in qa_pairs:
        with open(os.path.join(data_root, sample["api_response_path"]), "r", encoding="utf-8") as f:
            tmp = json.load(f)
            api_response = tmp[sample["app"]][sample["endpoint"]][sample["api_query"]]

        with open(os.path.join(data_root, sample["api_response_schema"]), "r", encoding="utf-8") as f:
            schema = f.read()

        if simplify_json:
            schema = json.loads(schema)
            api_response, schema = get_api_response_cf(api_response, schema, sample, task)
            schema = str(schema)

        out.append(LongResponseQASample(
            api_response=api_response,
            question=sample["question"],
            gold_answer=sample["gold_answer"],
            schema=schema,
            pred_answer=None,
            model_output=None,
            code_exec_status=None,
            metrics=sample["metrics"],
            task=sample["task"],
            task_type=sample["task_type"],
            uid=sample["uid"],
        ))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--setup", required=True,
                        choices=list(SETUP_ALIASES.keys()) + list(SETUP_ALIASES.values()),
                        help="Which baseline column to reproduce")
    parser.add_argument("--model", required=True,
                        help='Model id, e.g. "openai/gpt-oss-20b" or "ibm-granite/granite-3.3-8b-instruct"')
    parser.add_argument("--workers", type=int, default=8,
                        help="Worker processes for parallel inference (0 = serial)")
    parser.add_argument("--results-dir", default=os.path.join(ROOT, "experimental_scripts", "results"),
                        help="Where predictions get written")
    parser.add_argument("--data-root", default=os.path.join(ROOT, "..", "data", "toolJSONprocessing", "generate_qa_pairs", "data"),
                        help="Path to generate_qa_pairs/data/")
    parser.add_argument("--limit", type=int, default=None,
                        help="If set, only run the first N samples per endpoint (smoke test)")
    args = parser.parse_args()

    setup_type = SETUP_ALIASES.get(args.setup, args.setup)
    simplify_json = "cfx" in setup_type

    pred_dir = os.path.join(args.results_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    print("=" * 70)
    print(f"BASELINE | setup={setup_type} | model={args.model}")
    print("=" * 70)

    for task in TASK_LISTS:
        out_path = os.path.join(
            pred_dir,
            f"{task}_{args.model.split('/')[-1]}_{setup_type}_predictions.json",
        )
        if os.path.exists(out_path):
            print(f"  [skip] already exists: {os.path.basename(out_path)}")
            continue

        qa_file = os.path.join(args.data_root, "qa_pairs", task + "_qa_pairs.json")
        if not os.path.exists(qa_file):
            print(f"  [warn] no qa_pairs file: {qa_file}")
            continue

        qa_pair_objs = build_qa_pairs(qa_file, args.data_root, simplify_json, task)
        if args.limit is not None:
            qa_pair_objs = qa_pair_objs[: args.limit]

        print(f"  [run]  {task}  ({len(qa_pair_objs)} samples)")

        if args.workers <= 0:
            results = run_inference(qa_pair_objs, setup_type, args.model, DEFAULT_LLM_PARAMETERS)
        else:
            chunked = [([s], setup_type, args.model, dict(DEFAULT_LLM_PARAMETERS)) for s in qa_pair_objs]
            with Pool(processes=args.workers) as pool:
                output_lists = pool.starmap(run_inference, chunked)
            results = [s for batch in output_lists for s in batch]

        rows = []
        for s in results:
            rows.append({
                "endpoint": task,
                "setup_type": setup_type,
                "model": args.model,
                "uid": s.uid,
                "api_response": s.api_response,
                "question": s.question,
                "gold_answer": s.gold_answer,
                "schema": s.schema,
                "task": s.task,
                "task_type": s.task_type,
                "predicted_answer": s.pred_answer,
                "code_exec_status": s.code_exec_status,
                "model_output": s.model_output,
                "metrics": s.metrics,
            })
        with open(out_path, "w") as f:
            json.dump(rows, f)
        print(f"  [done] -> {out_path}")


if __name__ == "__main__":
    main()
