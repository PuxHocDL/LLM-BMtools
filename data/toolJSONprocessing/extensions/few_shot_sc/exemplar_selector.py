from dataclasses import dataclass
from typing import List, Literal, Optional
import numpy as np
import json

@dataclass
class Exemplar:
    question: str
    schema: str
    api_response_snippet: str
    code: str
    answer: str
    task_type: str
    template_id: str
    embedding: Optional[np.ndarray] = None

class ExemplarSelector:
    def __init__(self, pool: List[Exemplar], strategy: str = "retrieval"):
        self.pool = pool
        self.strategy = strategy

    def select(self, query: str, k: int, exclude_template: Optional[str], task_type: Optional[str] = None) -> List[Exemplar]:
        if k == 0:
            return []
        candidates = [e for e in self.pool if e.template_id != exclude_template]
        if task_type:
            same_task = [e for e in candidates if e.task_type == task_type]
            if same_task:
                candidates = same_task
        if not candidates:
            return []

        if self.strategy == "random":
            k_actual = min(k, len(candidates))
            return np.random.choice(candidates, k_actual, replace=False).tolist()
        elif self.strategy == "diverse":
            return self._mmr_select(candidates, k)
        elif self.strategy == "retrieval":
            return np.random.choice(candidates, min(k, len(candidates)), replace=False).tolist()
        return []

    def _mmr_select(self, cands, k, lambda_=0.5):
        if not cands: return []
        if len(cands) <= k: return cands
        
        # If no embeddings, just fallback to random
        if cands[0].embedding is None:
            return np.random.choice(cands, k, replace=False).tolist()

        selected = [cands[np.random.randint(len(cands))]]
        cands.remove(selected[0])

        while len(selected) < k and cands:
            best_score = -float('inf')
            best_idx = -1
            
            for i, cand in enumerate(cands):
                # We don't have query embedding here, so we maximize diversity among selected
                # diversity = min distance to already selected
                min_dist = float('inf')
                for sel in selected:
                    # Cosine distance = 1 - cosine similarity
                    sim = np.dot(cand.embedding, sel.embedding) / (np.linalg.norm(cand.embedding) * np.linalg.norm(sel.embedding) + 1e-9)
                    dist = 1.0 - sim
                    if dist < min_dist:
                        min_dist = dist
                
                score = min_dist
                if score > best_score:
                    best_score = score
                    best_idx = i
                    
            selected.append(cands[best_idx])
            cands.pop(best_idx)
            
        return selected

    def select_with_retrieval(self, query_embedding, k: int, exclude_template: Optional[str], task_type: Optional[str] = None) -> List[Exemplar]:
        candidates = [e for e in self.pool if e.template_id != exclude_template and e.embedding is not None]
        if task_type:
            same_task = [e for e in candidates if e.task_type == task_type]
            if same_task:
                candidates = same_task
        if not candidates or k == 0:
            return []

        scored = []
        for cand in candidates:
            sim = np.dot(query_embedding, cand.embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(cand.embedding) + 1e-9)
            scored.append((sim, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]
