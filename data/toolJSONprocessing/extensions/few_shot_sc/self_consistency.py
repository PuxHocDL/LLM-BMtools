from collections import Counter
import ast as py_ast

def sample_codes(model_client, prompt, K, temperature=0.7, top_p=0.95, stop=None, max_tokens=None, enable_thinking=False):
    """Generate K candidate responses."""
    codes = []
    for _ in range(K):
        codes.append(model_client.generate(
            prompt,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        ))
    return codes

def normalize_answer(a):
    if isinstance(a, str):
        return a.strip().lower()
    if isinstance(a, list):
        return tuple(sorted(map(str, a)))
    if isinstance(a, (float, int)):
        if isinstance(a, bool):
            return str(a).lower()
        return round(float(a), 4)
    return a

def vote(results, strategy: str = "answer-majority", **kwargs):
    """Aggregate K results into single answer."""
    valid = [a for a in results if a is not None and not (isinstance(a, str) and a.startswith("__ERROR"))]
    if not valid:
        return "__ERROR_NO_VALID_SAMPLES", 0.0
        
    def get_majority():
        normalized = [normalize_answer(a) for a in valid]
        counter = Counter(normalized)
        top, count = counter.most_common(1)[0]
        # Find original
        for original in valid:
            if normalize_answer(original) == top:
                return original, count / len(valid)
        return valid[0], 0.0

    if strategy == "answer-majority":
        return get_majority()
        
    elif strategy == "ast":
        # Simplified AST approach: fallback to majority if actual AST objects not provided
        # In a real AST setup, we'd hash the AST of the generated codes.
        return get_majority()
        
    elif strategy == "verifier":
        judge_client = kwargs.get('judge_client')
        schema_str = kwargs.get('schema_str', '{}')
        question = kwargs.get('question', '')
        if judge_client:
            from data.toolJSONprocessing.extensions.few_shot_sc.verifier import verifier_rerank
            ans = verifier_rerank(valid, schema_str, question, judge_client)
            return ans, 1.0 / len(valid)
        return get_majority()
        
    elif strategy == "hybrid":
        ans, conf = get_majority()
        if conf < 0.5: # If confidence is low, fallback to verifier
            return vote(results, strategy="verifier", **kwargs)
        return ans, conf
        
    return valid[0], 0.0
