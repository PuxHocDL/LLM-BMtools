"""Concurrent evaluation runner for the ToolJSON benchmark.

For each sample: build the prompt, generate a prediction (a single LLM call,
or -- when the loader exposes ``execute_debug_loop`` -- the self-correction
loop), then score it with EM / Contains / F1 / LLM-as-a-judge.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from core.llm_client import LLMClient
from core.metrics import Metrics

METRIC_KEYS = ("exact_match", "contains", "f1_score", "llm_judge")


class Evaluator:
    def __init__(self, data_loader, agent_name="qwen", judge_name="qwen"):
        self.data_loader = data_loader
        self.agent_name = agent_name
        self.client = LLMClient(agent_name=agent_name)
        self.judge_client = LLMClient(agent_name=judge_name)

    def _safe_write_json(self, path, payload):
        import json
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
            return path
        except OSError:
            fallback = f"{os.path.splitext(path)[0]}_{int(time.time() * 1000)}.json"
            with open(fallback, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
            return fallback

    def _process_single_sample(self, i, sample, skip_judge=False):
        prompt = self.data_loader.format_prompt(sample)
        ground_truth = self.data_loader.get_ground_truth(sample)
        question = self.data_loader.get_question(sample)

        start_time = time.time()
        tqdm.write(f"[Sample {i}] Starting... Q: {str(question)[:80]}")

        res = {
            "sample_id": i,
            "question/task": question,
            "ground_truth": ground_truth,
            "exact_match": 0,
            "contains": 0,
            "f1_score": 0.0,
            "llm_judge": -1 if skip_judge else 0,
            "error": None,
        }

        try:
            # Self-correction loaders drive their own generation loop; everything
            # else is a single LLM call followed by an optional postprocess hook.
            if hasattr(self.data_loader, "execute_debug_loop"):
                prediction = self.data_loader.execute_debug_loop(self.client, i, sample)
            else:
                prediction = self.client.generate(
                    prompt,
                    stop=self.data_loader.stop_sequences,
                    enable_thinking=getattr(self.data_loader, "enable_thinking", False),
                    max_tokens=getattr(self.data_loader, "max_tokens", None),
                )
                postprocess = getattr(self.data_loader, "postprocess_prediction", None)
                if callable(postprocess):
                    prediction = postprocess(prediction)

            res["prediction"] = prediction
            elapsed = time.time() - start_time

            if prediction == "__ERROR_CONTEXT_LENGTH__":
                res["error"] = "Context Length Exceeded"
                tqdm.write(f"[Sample {i}] ERROR: context length exceeded ({elapsed:.1f}s)")
                return res
            if isinstance(prediction, str) and prediction.startswith("__ERROR"):
                res["error"] = prediction
                tqdm.write(f"[Sample {i}] ERROR: generation failed ({elapsed:.1f}s)")
                return res

            res["exact_match"] = Metrics.exact_match(prediction, ground_truth)
            res["contains"] = Metrics.contains(prediction, ground_truth)
            res["f1_score"] = Metrics.f1_score(prediction, ground_truth)

            if skip_judge:
                res["llm_judge"] = -1
            else:
                try:
                    res["llm_judge"] = Metrics.llm_as_judge(
                        prediction, ground_truth, question, self.judge_client
                    )
                except Exception as e:
                    res["error"] = str(e)

            total_elapsed = time.time() - start_time
            tqdm.write(
                f"[Sample {i}] Done ({total_elapsed:.1f}s) | "
                f"EM={res['exact_match']} F1={res['f1_score']:.2f} Judge={res['llm_judge']}"
            )
        except Exception as e:
            elapsed = time.time() - start_time
            res["error"] = str(e)
            res["prediction"] = ""
            tqdm.write(f"[Sample {i}] EXCEPTION ({elapsed:.1f}s): {str(e)[:100]}")

        return res

    def evaluate(self, limit=None, output_dir="results", workers=1, skip_judge=False):
        os.makedirs(output_dir, exist_ok=True)
        data = self.data_loader.data
        total_samples = len(data)
        if limit:
            data = data[:limit]

        print("Dataset Statistics:")
        print(f" - Total Samples    : {total_samples}")
        print(f" - Samples Evaluated: {len(data)}")
        print(f" - Concurrent Workers: {workers}")
        print(f"Starting evaluation on {len(data)} samples using {self.agent_name}...")

        results = []
        trace_log = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._process_single_sample, i, sample, skip_judge): i
                for i, sample in enumerate(data)
            }
            total_timeout = max(300, (len(data) / max(workers, 1)) * 60)
            for future in tqdm(as_completed(futures, timeout=total_timeout), total=len(data)):
                try:
                    res = future.result(timeout=300)
                    if res.get("error"):
                        trace_log.append({
                            "sample_id": res["sample_id"],
                            "loader": self.data_loader.__class__.__name__,
                            "message": res["error"],
                        })
                    res.pop("error", None)
                    results.append(res)
                except TimeoutError:
                    idx = futures[future]
                    print(f"\nSample {idx} timed out, skipping...")
                    results.append({
                        "sample_id": idx, "question/task": "", "ground_truth": "",
                        "prediction": "", "exact_match": 0, "contains": 0,
                        "f1_score": 0.0, "llm_judge": -1 if skip_judge else 0,
                    })
                except Exception as e:
                    print(f"Fatal error processing sample: {e}")

        results.sort(key=lambda x: x["sample_id"])

        summary_metrics = {key: 0.0 for key in METRIC_KEYS}
        for res in results:
            for key in METRIC_KEYS:
                summary_metrics[key] += res[key]
        if results:
            for key in METRIC_KEYS:
                summary_metrics[key] /= len(results)

        loader_name = self.data_loader.__class__.__name__
        output_json = os.path.join(output_dir, f"{loader_name}_{self.agent_name}_detailed.json")
        output_csv = os.path.join(output_dir, f"{loader_name}_{self.agent_name}_detailed.csv")

        saved_json = self._safe_write_json(output_json, {"summary": summary_metrics, "details": results})
        try:
            pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8")
        except OSError:
            output_csv = os.path.join(
                output_dir, f"{loader_name}_{self.agent_name}_detailed_{int(time.time() * 1000)}.csv"
            )
            pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8")

        if trace_log:
            self._safe_write_json(os.path.join(output_dir, f"{loader_name}_trace.json"), trace_log)

        print("\nEvaluation Summary:")
        for key, value in summary_metrics.items():
            print(f"  {key}: {value:.4f}")
        print(f"\nResults saved to {output_dir}")
        if saved_json != output_json:
            print(f"Detailed JSON saved to {saved_json}")
        return summary_metrics
