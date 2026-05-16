import pandas as pd
import glob
import os
from core.metrics import Metrics

def evaluate_csv_files(input_dir="results"):
    # Find all CSV files in the input directory that end with _detailed.csv
    csv_files = glob.glob(os.path.join(input_dir, "*_detailed.csv"))
    
    if not csv_files:
        print(f"No *_detailed.csv files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files. Starting evaluation...\n")

    for file_path in csv_files:
        print(f"Evaluating: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path)
            
            # Create a dictionary to hold sums for averaging later
            metric_sums = {
                "exact_match": 0,
                "contains": 0,
                "action_match": 0,
                "f1_score": 0,
                "bleu_score": 0,
                "rouge1": 0,
                "rougeL": 0
            }
            
            valid_samples = len(df)
            if valid_samples == 0:
                print("  -> File is empty. Skipping.\n")
                continue

            for idx, row in df.iterrows():
                pred = str(row.get("prediction", ""))
                if pd.isna(row.get("prediction")):
                    pred = ""
                    
                gt = str(row.get("ground_truth", ""))
                if pd.isna(row.get("ground_truth")):
                    gt = ""

                # Compute metrics
                em = Metrics.exact_match(pred, gt)
                cont = Metrics.contains(pred, gt)
                act_m = Metrics.action_match(pred, gt)
                f1 = Metrics.f1_score(pred, gt)
                bleu = Metrics.bleu_score(pred, gt)
                
                # Rouge returns a dict
                rouge_scores = Metrics.rouge_score(pred, gt)
                r1 = rouge_scores.get('rouge1', 0.0)
                rl = rouge_scores.get('rougeL', 0.0)

                # Update DataFrame
                df.at[idx, "exact_match"] = em
                df.at[idx, "contains"] = cont
                df.at[idx, "action_match"] = act_m
                df.at[idx, "f1_score"] = f1
                df.at[idx, "bleu_score"] = bleu
                df.at[idx, "rouge1"] = r1
                df.at[idx, "rougeL"] = rl

                # Add to sums
                metric_sums["exact_match"] += em
                metric_sums["contains"] += cont
                metric_sums["action_match"] += act_m
                metric_sums["f1_score"] += f1
                metric_sums["bleu_score"] += bleu
                metric_sums["rouge1"] += r1
                metric_sums["rougeL"] += rl

            # Save the updated CSV
            df.to_csv(file_path, index=False, encoding="utf-8")
            
            # Print averages
            print(f"  --- Results for {os.path.basename(file_path)} ({valid_samples} samples) ---")
            for metric, total in metric_sums.items():
                avg = total / valid_samples
                print(f"  {metric:<15}: {avg:.4f}")
            print("  -> Updated CSV saved.\n")

        except Exception as e:
            print(f"  -> Error processing {file_path}: {e}\n")

if __name__ == "__main__":
    evaluate_csv_files("results")
