import pandas as pd
import glob
import os
from core.metrics import Metrics

def regrade_contains(input_dir="results"):
    csv_files = glob.glob(os.path.join(input_dir, "*_detailed.csv"))
    
    if not csv_files:
        print(f"No *_detailed.csv files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files. Recalculating 'contains' metric...\n")

    for file_path in csv_files:
        print(f"Evaluating 'contains' for: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path)
            
            contains_sum = 0
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

                # Compute ONLY the contains metric
                cont = Metrics.contains(pred, gt)
                df.at[idx, "contains"] = cont
                contains_sum += cont

            # Save the updated CSV
            df.to_csv(file_path, index=False, encoding="utf-8")
            
            # Print averages
            avg_contains = contains_sum / valid_samples
            print(f"  --- Results for {os.path.basename(file_path)} ({valid_samples} samples) ---")
            print(f"  contains avg   : {avg_contains:.4f}")
            print("  -> Updated CSV saved.\n")

        except Exception as e:
            print(f"  -> Error processing {file_path}: {e}\n")

if __name__ == "__main__":
    regrade_contains("results")
