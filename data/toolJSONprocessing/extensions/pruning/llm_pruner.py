import json
from .reconstruct import reconstruct

def get_skeleton(obj, depth=0, max_depth=2):
    if depth >= max_depth:
        return type(obj).__name__
    if isinstance(obj, dict):
        return {k: get_skeleton(v, depth + 1, max_depth) for k, v in list(obj.items())[:20]}
    elif isinstance(obj, list):
        return [get_skeleton(obj[0], depth + 1, max_depth)] if obj else []
    else:
        return type(obj).__name__

class LLMPruner:
    def __init__(self, llm_generate_fn):
        self.llm_generate_fn = llm_generate_fn

    def prune(self, json_obj, question):
        skeleton = get_skeleton(json_obj, max_depth=3)
        prompt = (
            "Given the following JSON schema/skeleton and a question, identify the relevant JSON paths needed to answer the question.\n"
            "Return ONLY a valid JSON array of strings representing the full dot-notation paths (e.g. [\"data.rooms\", \"data.hotel_info\"]).\n"
            f"Schema:\n{json.dumps(skeleton, indent=2)}\n\n"
            f"Question:\n{question}\n"
        )
        try:
            resp = self.llm_generate_fn(prompt)
            start = resp.find("[")
            end = resp.rfind("]")
            if start != -1 and end != -1 and end > start:
                paths = json.loads(resp[start:end+1])
                if isinstance(paths, list):
                    return reconstruct(json_obj, paths)
        except Exception as e:
            pass
        return json_obj
