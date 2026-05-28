import os
import json
import numpy as np
from typing import List, Dict
from data.toolJSONprocessing.extensions.few_shot_sc.exemplar_selector import Exemplar

class ExemplarPoolBuilder:
    def __init__(self, qa_data_dir: str, cache_path: str = "exemplars_cache.json", embedding_fn=None):
        self.qa_data_dir = qa_data_dir
        self.cache_path = cache_path
        self.embedding_fn = embedding_fn

    def build_pool(self) -> List[Exemplar]:
        if os.path.exists(self.cache_path):
            return self._load_from_cache()

        pool = []
        # Fallback to generating dummy pool if cache doesn't exist
        print(f"Cache {self.cache_path} not found. Building exemplar pool...")
        if os.path.isdir(self.qa_data_dir):
            for filename in os.listdir(self.qa_data_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.qa_data_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Just take first few valid examples per endpoint
                        count = 0
                        for item in data:
                            if count >= 10: # Limit per endpoint
                                break
                            if item.get("gold_answer") is not None:
                                # Create dummy code for exemplar
                                ans = item.get("gold_answer")
                                code_str = f"def extract_data(data):\n    return {repr(ans)}"
                                
                                ex = Exemplar(
                                    question=item.get("question", ""),
                                    schema=item.get("api_response_schema", "{}"),
                                    api_response_snippet="{}",
                                    code=code_str,
                                    answer=str(ans),
                                    task_type=item.get("task_type", ""),
                                    template_id=item.get("task", "")
                                )
                                pool.append(ex)
                                count += 1
                                
        if self.embedding_fn and pool:
            texts = [e.question for e in pool]
            embeddings = self.embedding_fn(texts)
            for e, emb in zip(pool, embeddings):
                e.embedding = np.array(emb)
                
        self._save_to_cache(pool)
        return pool

    def _save_to_cache(self, pool: List[Exemplar]):
        data = []
        for e in pool:
            data.append({
                "question": e.question,
                "schema": e.schema,
                "api_response_snippet": e.api_response_snippet,
                "code": e.code,
                "answer": e.answer,
                "task_type": e.task_type,
                "template_id": e.template_id,
                "embedding": e.embedding.tolist() if e.embedding is not None else None
            })
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _load_from_cache(self) -> List[Exemplar]:
        pool = []
        with open(self.cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                ex = Exemplar(
                    question=item["question"],
                    schema=item["schema"],
                    api_response_snippet=item["api_response_snippet"],
                    code=item["code"],
                    answer=item["answer"],
                    task_type=item["task_type"],
                    template_id=item["template_id"],
                    embedding=np.array(item["embedding"]) if item.get("embedding") else None
                )
                pool.append(ex)
        return pool
