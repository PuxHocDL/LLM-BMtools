import pandas as pd
import glob
import os

def summarize_results(input_dir="results"):
    csv_files = glob.glob(os.path.join(input_dir, "*_detailed.csv"))
    
    # Model name mapping based on filenames
    model_mapping = {
        "devstral": "mistralai/Devstral-Small-2507",
        "gptoss": "gpt-oss-20b",
        "granite": "granite-33-8b"
    }

    results = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path)
        
        # Determine model name
        model_name = filename.replace("HeuristicPlusCodeSchema_ToolJSON_", "").replace("HeuristicPlus_ToolJSON_", "").replace("_detailed.csv", "")
        if model_name in model_mapping:
            model_name = model_mapping[model_name]
            
        # Add strategy if CodeSchema is used
        if "CodeSchema" in filename:
            model_name = f"{model_name} (CodeSchema)"

        total_samples = len(df)
        if total_samples == 0:
            continue

        # Averages
        em = df["exact_match"].mean() if "exact_match" in df.columns else 0
        f1 = df["f1_score"].mean() if "f1_score" in df.columns else 0
        contains = df["contains"].mean() if "contains" in df.columns else 0
        
        # Judge (only average valid scores >= 0)
        if "llm_judge" in df.columns:
            valid_judge = df[df["llm_judge"] >= 0]
            judge = valid_judge["llm_judge"].mean() if not valid_judge.empty else 0
        else:
            judge = 0

        # Errors and Empty
        errors_count = 0
        empty_count = 0
        
        for pred in df["prediction"]:
            if pd.isna(pred) or str(pred).strip() == "":
                empty_count += 1
            elif "ERROR" in str(pred).upper():
                errors_count += 1

        results.append({
            "model": model_name,
            "EM": f"{em:.4f}",
            "F1": f"{f1:.4f}",
            "Contains": f"{contains:.4f}",
            "Judge": f"{judge:.4f}",
            "Errors": str(errors_count),
            "Empty": str(empty_count)
        })

    # Print TSV format
    print("model\tEM\tF1\tContains\tJudge\tErrors\tEmpty")
    for r in results:
        print(f"{r['model']}\t{r['EM']}\t{r['F1']}\t{r['Contains']}\t{r['Judge']}\t{r['Errors']}\t{r['Empty']}")

if __name__ == "__main__":
    summarize_results("results")
