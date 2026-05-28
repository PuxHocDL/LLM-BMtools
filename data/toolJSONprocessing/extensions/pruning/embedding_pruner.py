import numpy as np
from .flattening import flatten
from .reconstruct import reconstruct

class EmbeddingPruner:
    def __init__(self, embed_fn, top_k_paths=20):
        self.embed_fn = embed_fn
        self.top_k_paths = top_k_paths

    def prune(self, json_obj, question):
        flat = flatten(json_obj)
        if not flat:
            return json_obj

        paths = list(flat.keys())
        texts_to_embed = [f"{p}: {flat[p]}" for p in paths]
        
        try:
            all_embeddings = self.embed_fn([question] + texts_to_embed)
            q_emb = np.array(all_embeddings[0])
            p_embs = np.array(all_embeddings[1:])
            
            q_norm = np.linalg.norm(q_emb)
            p_norms = np.linalg.norm(p_embs, axis=1)
            
            sims = np.dot(p_embs, q_emb) / (np.maximum(p_norms * q_norm, 1e-9))
            
            top_k = min(self.top_k_paths, len(paths))
            top_indices = np.argsort(sims)[::-1][:top_k]
            
            keep_paths = {paths[i] for i in top_indices}
            return reconstruct(json_obj, keep_paths)
        except Exception as e:
            return json_obj
