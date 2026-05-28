def verifier_rerank(answers, schema_str, question, judge_client):
    """Use an LLM to rerank answers and select the most plausible one."""
    if not answers:
        return None
    
    unique_ans = list(set(answers))
    if len(unique_ans) == 1:
        return unique_ans[0]
        
    prompt = f"Given the following JSON schema:\n{schema_str}\n\n"
    prompt += f"And the question: {question}\n\n"
    prompt += "Which of the following answers is the most plausible and correct based on the schema and question context?\n"
    for i, ans in enumerate(unique_ans):
        prompt += f"Answer {i+1}: {ans}\n"
        
    prompt += "\nOutput ONLY the number of the best answer (e.g., 1 or 2)."
    
    response = judge_client.generate(prompt, max_tokens=10)
    try:
        best_idx = int(''.join(filter(str.isdigit, response))) - 1
        if 0 <= best_idx < len(unique_ans):
            return unique_ans[best_idx]
    except Exception:
        pass
    
    return unique_ans[0] # Fallback to first if parsing fails
