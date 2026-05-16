import argparse
import os
import json
import time
import threading
from data_loaders.tooljson import ToolJSONLoader
from evaluators.evaluator import Evaluator
from core.llm_client import LLMClient
from openai import OpenAI, RateLimitError

from data.toolJSONprocessing.extensions.pruning.heuristic_plus import heuristic_plus_prune
from data.toolJSONprocessing.extensions.pruning.embedding_pruner import EmbeddingPruner
from data.toolJSONprocessing.extensions.pruning.llm_pruner import LLMPruner
from data.toolJSONprocessing.extensions.pruning.coverage_eval import coverage

HEURISTIC_PLUS_METHODS = {"heuristic_plus", "heuristic_plus_code_schema"}


def _merge_schema(left, right):
    if left == right:
        return left
    if not isinstance(left, dict) or not isinstance(right, dict):
        return sorted({str(left), str(right)})
    if left.get("type") != right.get("type"):
        return {"type": sorted({str(left.get("type")), str(right.get("type"))})}
    if left.get("type") == "object":
        keys = sorted(set(left.get("properties", {})) | set(right.get("properties", {})))
        return {
            "type": "object",
            "properties": {
                key: _merge_schema(
                    left.get("properties", {}).get(key, {"type": "missing"}),
                    right.get("properties", {}).get(key, {"type": "missing"})
                )
                for key in keys
            }
        }
    if left.get("type") == "array":
        return {
            "type": "array",
            "items": _merge_schema(left.get("items", {}), right.get("items", {}))
        }
    return left


def _infer_compact_schema(value, depth=0, max_depth=4, max_list_items=4):
    if depth >= max_depth:
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {
                str(key): _infer_compact_schema(child, depth + 1, max_depth, max_list_items)
                for key, child in value.items()
            }
        }
    if isinstance(value, list):
        if not value:
            return {"type": "array", "items": {"type": "unknown"}}
        item_schema = _infer_compact_schema(value[0], depth + 1, max_depth, max_list_items)
        for item in value[1:max_list_items]:
            item_schema = _merge_schema(item_schema, _infer_compact_schema(item, depth + 1, max_depth, max_list_items))
        return {"type": "array", "items": item_schema}
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    return {"type": "string"}


def _format_answer_prompt(question, pruned_json_str, summary_answer):
    prompt = "Analyze the following JSON output to answer the user's question.\n"
    prompt += "If the JSON contains __question_summary__.answer, return exactly that value and nothing else.\n"
    prompt += "Otherwise, answer with only the requested value, count, or comma-separated list. Do not explain.\n"
    prompt += f"JSON Output:\n{pruned_json_str}\n\n"
    prompt += f"Question:\n{question}\n"
    if summary_answer is not None:
        prompt += f"Extracted answer candidate:\n{summary_answer}\n"
        prompt += "Return exactly the extracted answer candidate and nothing else."
    else:
        prompt += "Answer directly based on the JSON content."
    return prompt


def _format_code_schema_prompt(question, pruned_obj, pruned_json_str, summary_answer):
    schema = _infer_compact_schema(pruned_obj)
    schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
    prompt = (
        "You are solving a data task using Python-style computation over a pruned API response.\n"
        "Use the schema to identify the relevant fields, then apply exact filtering, counting, sorting, "
        "aggregation, or list construction as if `data` were the JSON object below.\n"
        "Do not guess from field names alone; compute from the provided values.\n"
        "Return ONLY the final answer value, count, date, name, or comma-separated list. No explanation.\n\n"
        "Data schema:\n"
        f"{schema_str}\n\n"
        "Pruned JSON assigned to variable `data`:\n"
        f"{pruned_json_str}\n\n"
        f"Question:\n{question}\n\n"
    )
    if summary_answer is not None:
        prompt += (
            "A deterministic preprocessing step already computed this candidate:\n"
            f"{summary_answer}\n"
            "Verify it against the JSON/schema. If consistent, output exactly the candidate. "
            "If inconsistent, compute the corrected answer from `data`."
        )
    else:
        prompt += (
            "Think in terms of a short Python snippet over `data`, but do not output the snippet. "
            "Output only the computed answer."
        )
    return prompt


class PrunedToolJSONLoader(ToolJSONLoader):
    stop_sequences = ["\n\nQuestion:"]
    enable_thinking = False
    max_tokens = 1000

    def __init__(self, data_path, base_repo_dir="data/toolJSONprocessing", agent_name=None, 
                 pruning_method=None, budget=10000, pruner_obj=None, gt_paths_dict=None):
        super().__init__(data_path, base_repo_dir, agent_name)
        self.pruning_method = pruning_method
        self.budget = budget
        self.pruner_obj = pruner_obj
        self.gt_paths_dict = gt_paths_dict
        
        # Stats
        self.total_original_chars = 0
        self.total_pruned_chars = 0
        self.total_pruning_time = 0
        self.total_coverage = 0
        self.samples_processed = 0
        self._stats_lock = threading.Lock()

    def format_prompt(self, sample):
        question = sample.get("question", "")
        api_response_path = sample.get("api_response_path", "")
        
        filename = os.path.basename(api_response_path)
        api_res_full_path = os.path.join(self.base_repo_dir, "generate_qa_pairs", "data", "api_responses", filename)
        api_res_full_path = api_res_full_path.replace("?", "_").replace(":", "_").replace('"', "_")
        
        json_content_str = ""
        try:
            with open(api_res_full_path, 'r', encoding='utf-8') as f:
                json_content_str = f.read()
                json_obj = json.loads(json_content_str)
        except Exception as e:
            return super().format_prompt(sample)

        start_time = time.time()
        
        gt_paths = []
        if self.gt_paths_dict:
            task_name = sample.get("task", "")
            if task_name in self.gt_paths_dict:
                gt_paths.extend(self.gt_paths_dict[task_name])

        pruned_obj = json_obj
        if self.pruning_method in HEURISTIC_PLUS_METHODS:
            pruned_obj = heuristic_plus_prune(
                json_obj,
                question,
                max_chars=self.budget,
                query_context=sample.get("api_query")
            )
        elif self.pruning_method == "embedding" and self.pruner_obj:
            pruned_obj = self.pruner_obj.prune(json_obj, question)
        elif self.pruning_method == "llm_guided" and self.pruner_obj:
            pruned_obj = self.pruner_obj.prune(json_obj, question)

        pruned_json_str = json.dumps(pruned_obj, indent=2)
        
        if self.pruning_method not in HEURISTIC_PLUS_METHODS:
            if len(pruned_json_str) < 100 or len(pruned_json_str) > self.budget:
                pruned_obj = heuristic_plus_prune(
                    json_obj,
                    question,
                    max_chars=self.budget,
                    query_context=sample.get("api_query")
                )
                pruned_json_str = json.dumps(pruned_obj, indent=2)

        prune_time = time.time() - start_time
        
        cov = coverage(pruned_obj, gt_paths) if gt_paths else None
        with self._stats_lock:
            self.total_original_chars += len(json_content_str)
            self.total_pruned_chars += len(pruned_json_str)
            self.total_pruning_time += prune_time
            if cov is not None:
                self.total_coverage += cov
            self.samples_processed += 1
        
        summary_answer = None
        if isinstance(pruned_obj, dict):
            summary_answer = pruned_obj.get("__question_summary__", {}).get("answer")

        if self.pruning_method == "heuristic_plus_code_schema":
            prompt = _format_code_schema_prompt(question, pruned_obj, pruned_json_str, summary_answer)
        else:
            prompt = _format_answer_prompt(question, pruned_json_str, summary_answer)
        
        audit_dir = "results/ablation_3_audits"
        os.makedirs(audit_dir, exist_ok=True)
        audit_name = str(sample.get("uid") or f"sample_{self.samples_processed}")
        audit_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in audit_name)
        audit_payload = {
            "question": question,
            "original_chars": len(json_content_str),
            "pruned_chars": len(pruned_json_str),
            "coverage": cov,
            "pruned_json": pruned_obj
        }
        audit_path = f"{audit_dir}/{self.pruning_method}_{audit_name}.json"
        try:
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(audit_payload, f, indent=2)
        except OSError:
            fallback_path = f"{audit_dir}/{self.pruning_method}_{audit_name}_{int(time.time() * 1000)}.json"
            try:
                with open(fallback_path, "w", encoding="utf-8") as f:
                    json.dump(audit_payload, f, indent=2)
            except OSError:
                pass

        return [{"role": "user", "content": prompt}]

class HeuristicPlus_ToolJSON(PrunedToolJSONLoader):
    pass

class HeuristicPlusCodeSchema_ToolJSON(PrunedToolJSONLoader):
    pass

class Embedding_ToolJSON(PrunedToolJSONLoader):
    pass

class LLMGuided_ToolJSON(PrunedToolJSONLoader):
    pass

def get_openai_embedding_fn(api_key):
    client = OpenAI(api_key=api_key)
    def embed_fn(texts):
        max_retries = 5
        for i in range(max_retries):
            try:
                response = client.embeddings.create(
                    input=texts,
                    model="text-embedding-3-small"
                )
                return [data.embedding for data in response.data]
            except RateLimitError as e:
                if i == max_retries - 1:
                    raise e
                wait_time = (2 ** i) + 2
                print(f"  [RateLimit] Waiting {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"  [Embedding Error] {e}")
                time.sleep(2)
        return []
    return embed_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen", help="Agent model from config")
    parser.add_argument("--judge", type=str, default="qwen", help="Judge model from config")
    parser.add_argument("--method", type=str, required=True, choices=["heuristic_plus", "heuristic_plus_code_schema", "embedding", "llm_guided"])
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--skip_judge", action="store_true", default=True, help="Skip LLM-as-a-judge (Phase 1)")
    args = parser.parse_args()

    data_path = "data/toolJSONprocessing/generate_qa_pairs/data/qa_pairs"
    
    gt_paths_dict = {}
    gt_file = "data/toolJSONprocessing/groundtruth_paths.json"
    if os.path.exists(gt_file):
        with open(gt_file, "r", encoding="utf-8") as f:
            gt_paths_dict = json.load(f)

    pruner_obj = None
    if args.method == "embedding":
        import yaml
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        openai_key = cfg["agents"]["gpt-4o"]["api_key"]
        embed_fn = get_openai_embedding_fn(openai_key)
        pruner_obj = EmbeddingPruner(embed_fn=embed_fn, top_k_paths=20)
        
    elif args.method == "llm_guided":
        helper_client = LLMClient(agent_name=args.model)
        def llm_generate_fn(prompt):
            resp = helper_client.generate(prompt, stop=None, max_tokens=1000)
            if not resp:
                return "[]"
            return resp
        pruner_obj = LLMPruner(llm_generate_fn=llm_generate_fn)

    loader_cls = {
        "heuristic_plus": HeuristicPlus_ToolJSON,
        "heuristic_plus_code_schema": HeuristicPlusCodeSchema_ToolJSON,
        "embedding": Embedding_ToolJSON,
        "llm_guided": LLMGuided_ToolJSON
    }.get(args.method, PrunedToolJSONLoader)

    loader = loader_cls(
        data_path=data_path,
        agent_name=args.model,
        pruning_method=args.method,
        budget=args.budget,
        pruner_obj=pruner_obj,
        gt_paths_dict=gt_paths_dict
    )

    evaluator = Evaluator(data_loader=loader, agent_name=args.model, judge_name=args.judge)
    print(f"\n[Running Ablation 3 (Phase 1): {args.method.upper()} (Budget: {args.budget} chars)]")
    print(f"Dataset: {len(loader.data)} samples total.")
    evaluator.evaluate(limit=args.limit, workers=args.workers, skip_judge=args.skip_judge)
    
    if loader.samples_processed > 0:
        print("\n--- Pruning Metrics ---")
        avg_orig = loader.total_original_chars / loader.samples_processed
        avg_pruned = loader.total_pruned_chars / loader.samples_processed
        avg_time = loader.total_pruning_time / loader.samples_processed
        avg_cov = loader.total_coverage / loader.samples_processed
        print(f"Avg Original Chars : {avg_orig:.1f}")
        print(f"Avg Pruned Chars   : {avg_pruned:.1f} ({(avg_pruned/avg_orig)*100:.1f}%)")
        print(f"Avg Pruning Time   : {avg_time:.3f} s")
        print(f"Avg Path Coverage  : {avg_cov*100:.1f}%")

if __name__ == "__main__":
    main()
