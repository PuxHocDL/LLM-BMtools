import os
import json
import time
import pandas as pd
from tqdm import tqdm
from core.llm_client import LLMClient
from core.metrics import Metrics

class Evaluator:
    def __init__(self, data_loader, agent_name="qwen", judge_name="qwen"):
        self.data_loader = data_loader
        self.agent_name = agent_name
        self.client = LLMClient(agent_name=agent_name)
        self.judge_client = self.client if judge_name == agent_name else LLMClient(agent_name=judge_name)

    def _process_single_sample(self, i, sample):
        prompt = self.data_loader.format_prompt(sample)
        ground_truth = self.data_loader.get_ground_truth(sample)
        question = self.data_loader.get_question(sample)
        
        start_time = time.time()
        tqdm.write(f"[Sample {i}] Starting... Q: {str(question)[:80]}")
        
        # Base result
        res = {
            "sample_id": i,
            "question/task": question,
            "ground_truth": ground_truth,
            "exact_match": 0, "contains": 0, "action_match": 0,
            "f1_score": 0, "bleu_score": 0, "rouge1": 0, "rougeL": 0, "llm_judge": 0,
            "error": None
        }

        try:
            # Predict (use loader-specific stop sequences and thinking mode)
            stop = self.data_loader.stop_sequences
            enable_thinking = getattr(self.data_loader, 'enable_thinking', False)
            prediction = self.client.generate(prompt, stop=stop, enable_thinking=enable_thinking)
            res["prediction"] = prediction
            elapsed = time.time() - start_time

            if prediction == "__ERROR_CONTEXT_LENGTH__":
                res["error"] = "Context Length Exceeded"
                tqdm.write(f"[Sample {i}] ERROR: Context length exceeded ({elapsed:.1f}s)")
                return res
            elif prediction.startswith("__ERROR_API__"):
                res["error"] = prediction
                tqdm.write(f"[Sample {i}] ERROR: API error ({elapsed:.1f}s)")
                return res

            tqdm.write(f"[Sample {i}] LLM responded ({elapsed:.1f}s), computing metrics...")

            # Compute metrics
            res["exact_match"] = Metrics.exact_match(prediction, ground_truth)
            res["contains"] = Metrics.contains(prediction, ground_truth)
            res["f1_score"] = Metrics.f1_score(prediction, ground_truth)
            res["bleu_score"] = Metrics.bleu_score(prediction, ground_truth)
            rouge = Metrics.rouge_score(prediction, ground_truth)
            res["rouge1"] = rouge["rouge1"]
            res["rougeL"] = rouge["rougeL"]
            res["action_match"] = Metrics.action_match(prediction, ground_truth)
            
            judge_pred = prediction[:1000] if len(prediction) > 1000 else prediction
            judge_gt = ground_truth[:500] if len(ground_truth) > 500 else ground_truth
            judge_q = question[:500] if len(question) > 500 else question
            for judge_attempt in range(2):
                try:
                    res["llm_judge"] = Metrics.llm_as_judge(judge_pred, judge_gt, judge_q, self.judge_client)
                    break
                except Exception as e:
                    import traceback as tb
                    tqdm.write(f"[Sample {i}] Judge attempt {judge_attempt+1} failed: {type(e).__name__}: {str(e)[:150]}")
                    if judge_attempt == 0:
                        tb.print_exc()
                    if judge_attempt == 1:
                        res["error"] = f"LLM Judge failed: {e}"
                    else:
                        time.sleep(2)
            
            total_elapsed = time.time() - start_time
            tqdm.write(f"[Sample {i}] Done ({total_elapsed:.1f}s) | EM={res['exact_match']} F1={res['f1_score']:.2f} Judge={res['llm_judge']}")
                
        except Exception as e:
            elapsed = time.time() - start_time
            res["error"] = str(e)
            res["prediction"] = ""
            tqdm.write(f"[Sample {i}] EXCEPTION ({elapsed:.1f}s): {str(e)[:100]}")

        return res

    def evaluate(self, limit=None, output_dir="results", workers=1):
        os.makedirs(output_dir, exist_ok=True)
        data = self.data_loader.data
        
        total_samples = len(data)
        if limit:
            data = data[:limit]

        results = []
        trace_log = []
        summary_metrics = {
            "exact_match": 0, "contains": 0, "f1_score": 0, "bleu_score": 0,
            "rouge1": 0, "rougeL": 0, "action_match": 0, "llm_judge": 0
        }

        print(f"Dataset Statistics:")
        print(f" - Total Samples: {total_samples}")
        print(f" - Limit Applied: {len(data)}")
        print(f" - Concurrent Workers: {workers}")
        print(f"Starting evaluation on {len(data)} samples using {self.agent_name}...")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._process_single_sample, i, sample): i for i, sample in enumerate(data)}
            total_timeout = max(600, (len(data) / max(workers, 1)) * 300)
            for future in tqdm(as_completed(futures, timeout=total_timeout), total=len(data)):
                try:
                    res = future.result(timeout=30)
                    
                    if res.get("error"):
                        trace_log.append({
                            "sample_id": res["sample_id"],
                            "dataset": self.data_loader.__class__.__name__,
                            "error_type": "Evaluation Error",
                            "message": res["error"]
                        })
                        # Xoá trường error khỏi result chính để file CSV sạch sẽ
                        del res["error"]
                        results.append(res)
                    else:
                        if "error" in res:
                            del res["error"]
                        results.append(res)
                except TimeoutError:
                    idx = futures[future]
                    print(f"\nSample {idx} timed out, skipping...")
                    results.append({
                        "sample_id": idx,
                        "question/task": "",
                        "ground_truth": "",
                        "prediction": "",
                        "exact_match": 0, "contains": 0, "action_match": 0,
                        "f1_score": 0, "bleu_score": 0, "rouge1": 0, "rougeL": 0, "llm_judge": 0
                    })
                except Exception as e:
                    print(f"Fatal error processing sample: {e}")

        # Sắp xếp lại kết quả theo sample_id do chạy song song sẽ làm lộn xộn thứ tự
        results.sort(key=lambda x: x["sample_id"])

        # Aggregate metrics
        for res in results:
            summary_metrics["exact_match"] += res["exact_match"]
            summary_metrics["contains"] += res["contains"]
            summary_metrics["f1_score"] += res["f1_score"]
            summary_metrics["bleu_score"] += res["bleu_score"]
            summary_metrics["rouge1"] += res["rouge1"]
            summary_metrics["rougeL"] += res["rougeL"]
            summary_metrics["action_match"] += res["action_match"]
            summary_metrics["llm_judge"] += res["llm_judge"]

        # Calculate averages
        num_samples = len(results)
        if num_samples > 0:
            for k in summary_metrics:
                summary_metrics[k] /= num_samples

        # Save Detailed Results
        dataset_name = self.data_loader.__class__.__name__
        output_file_json = os.path.join(output_dir, f"{dataset_name}_{self.agent_name}_detailed.json")
        output_file_csv = os.path.join(output_dir, f"{dataset_name}_{self.agent_name}_detailed.csv")
        trace_file_json = os.path.join(output_dir, f"{dataset_name}_trace.json")
        
        with open(output_file_json, 'w', encoding='utf-8') as f:
            json.dump({"summary": summary_metrics, "details": results}, f, indent=4, ensure_ascii=False)
            
        df = pd.DataFrame(results)
        df.to_csv(output_file_csv, index=False, encoding='utf-8')

        if trace_log:
            with open(trace_file_json, 'w', encoding='utf-8') as f:
                json.dump(trace_log, f, indent=4, ensure_ascii=False)
            print(f"\nTrace log saved to {trace_file_json}")
        
        print("\nEvaluation Summary:")
        for k, v in summary_metrics.items():
            print(f"{k}: {v:.4f}")
            
        print(f"\nResults saved to {output_dir}")
        return summary_metrics
