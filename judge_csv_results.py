import argparse
import os
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from core.llm_client import LLMClient
from core.metrics import Metrics

def process_single_judge(i, row, judge_client):
    try:
        # Skip if already evaluated (not -1 and not empty)
        current_score = row.get("llm_judge", -1)
        if pd.notna(current_score) and current_score != -1:
            return i, current_score

        question = str(row.get("question/task", ""))
        ground_truth = str(row.get("ground_truth", ""))
        prediction = str(row.get("prediction", ""))
        
        # Nếu prediction rỗng hoặc bị lỗi
        if not prediction or "ERROR" in prediction or pd.isna(row.get("prediction")):
            return i, 0

        score = Metrics.llm_as_judge(prediction, ground_truth, question, judge_client)
        return i, score
    except Exception as e:
        print(f"Error on sample {i}: {e}")
        return i, -1

def evaluate_csv_llm_judge(input_dir, judge_name, workers):
    csv_files = glob.glob(os.path.join(input_dir, "*_detailed.csv"))
    
    if not csv_files:
        print(f"No *_detailed.csv files found in {input_dir}")
        return

    print(f"Initializing LLM Judge Agent: {judge_name}...")
    judge_client = LLMClient(agent_name=judge_name)

    for file_path in csv_files:
        print(f"\nEvaluating LLM Judge for: {os.path.basename(file_path)}")
        df = pd.read_csv(file_path)
        
        if "llm_judge" not in df.columns:
            df["llm_judge"] = -1
            
        rows_to_process = df.to_dict('records')
        updated_scores = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Gửi các task cần chấm (-1) vào executor
            futures = {}
            for i, row in enumerate(rows_to_process):
                if pd.isna(row.get("llm_judge")) or row.get("llm_judge", -1) == -1:
                    futures[executor.submit(process_single_judge, i, row, judge_client)] = i

            if not futures:
                print("  -> All samples already have llm_judge scores. Skipping.")
                continue

            # Sử dụng tqdm để hiển thị progress bar cho file hiện tại
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx = futures[future]
                try:
                    i, score = future.result()
                    if df.at[i, "llm_judge"] != score:
                        df.at[i, "llm_judge"] = score
                        if score != -1:
                            updated_scores += 1
                except Exception as e:
                    print(f"Failed to process future {idx}: {e}")

        # Tính lại trung bình và lưu file
        valid_samples = df[df["llm_judge"] != -1]
        avg_judge = valid_samples["llm_judge"].mean() if not valid_samples.empty else 0
        
        df.to_csv(file_path, index=False, encoding="utf-8")
        print(f"  -> Updated {updated_scores} samples.")
        print(f"  -> New LLM-as-a-judge Avg Score: {avg_judge:.4f}")
        print(f"  -> Saved {os.path.basename(file_path)}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM-as-a-judge for CSV files.")
    parser.add_argument("--dir", type=str, default="results", help="Directory containing CSV files")
    parser.add_argument("--judge", type=str, default="llama-70b", help="Agent name in config.yaml to use as Judge")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent threads")
    
    args = parser.parse_args()
    evaluate_csv_llm_judge(args.dir, args.judge, args.workers)

if __name__ == "__main__":
    main()
