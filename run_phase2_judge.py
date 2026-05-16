import argparse
import json
import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from core.llm_client import LLMClient
from core.metrics import Metrics

def process_single_judge(i, row, judge_client):
    try:
        # Check if it already has a valid score (1 or 0)
        # Skip if it's already evaluated (not -1)
        if row.get("llm_judge", -1) != -1:
            return i, row.get("llm_judge", 0)

        # Trích xuất thông tin
        question = row.get("question/task", "")
        ground_truth = row.get("ground_truth", "")
        prediction = row.get("prediction", "")
        
        # Nếu prediction rỗng hoặc bị lỗi, thường cho điểm 0 luôn hoặc vẫn đưa cho Judge
        if not prediction or "ERROR" in str(prediction):
            return i, 0

        score = Metrics.llm_as_judge(prediction, ground_truth, question, judge_client)
        return i, score
    except Exception as e:
        print(f"Error on sample {i}: {e}")
        return i, -1  # Retain -1 if failed

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Evaluate Phase 1 output using LLM-as-a-judge.")
    parser.add_argument("--input", type=str, required=True, help="Path to the JSON output from Phase 1")
    parser.add_argument("--judge", type=str, default="gpt-4o", help="Agent name in config.yaml to use as Judge")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent judge threads")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        return

    # Load file
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "details" not in data or "summary" not in data:
        print("Invalid input format. Expected a JSON with 'summary' and 'details' keys.")
        return

    details = data["details"]
    print(f"Loaded {len(details)} samples from {args.input}")

    # Initialize Judge Client
    print(f"Initializing LLM Judge: {args.judge}...")
    judge_client = LLMClient(agent_name=args.judge)

    print(f"Starting Phase 2 Evaluation with {args.workers} workers...")
    
    evaluated_count = 0
    updated_scores = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_judge, i, row, judge_client): i for i, row in enumerate(details)}
        
        for future in tqdm(as_completed(futures), total=len(details)):
            idx = futures[future]
            try:
                i, score = future.result()
                # Update row
                if details[i].get("llm_judge") != score:
                    details[i]["llm_judge"] = score
                    updated_scores += 1
                evaluated_count += 1
            except Exception as e:
                print(f"Failed to process future {idx}: {e}")

    # Recompute summary llm_judge score
    total_judge_score = sum(max(0, row.get("llm_judge", 0)) for row in details if row.get("llm_judge", -1) != -1)
    valid_samples = sum(1 for row in details if row.get("llm_judge", -1) != -1)
    
    avg_judge = total_judge_score / valid_samples if valid_samples > 0 else 0
    data["summary"]["llm_judge"] = avg_judge
    
    print(f"\nEvaluation Completed!")
    print(f"Updated {updated_scores} samples.")
    print(f"New LLM-as-a-judge Avg Score: {avg_judge:.4f}")

    # Save JSON
    output_json = args.input
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    # Save CSV
    output_csv = args.input.replace(".json", ".csv")
    df = pd.DataFrame(details)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Saved updated results to {output_json} and {output_csv}")

if __name__ == "__main__":
    main()
