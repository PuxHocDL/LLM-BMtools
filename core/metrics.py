import re
from collections import Counter
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

class Metrics:
    @staticmethod
    def _strip_thinking(text):
        """Remove <think>...</think> blocks and inline thinking from model output."""
        if not isinstance(text, str):
            return str(text)
        # 1. Remove <think>...</think> XML blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        # 2. Remove inline "Thinking Process:" blocks (Qwen-style)
        #    Only strip if there's a clear answer boundary after it
        answer_patterns = [
            r'(?:^|\n)\s*(?:ANSWER|Final Answer)\s*[:=]',
            r'(?:^|\n)\s*(?:Therefore|Thus|Hence|So)[,:]?\s+(?:the answer is|it is)',
        ]
        for ap in answer_patterns:
            m = re.search(ap, text, re.IGNORECASE)
            if m:
                # Find where the thinking block starts (before the answer)
                think_start = re.search(
                    r'(?i)(?:thinking process|let me think|my reasoning)[:\s]*\n',
                    text
                )
                if think_start and think_start.start() < m.start():
                    text = text[:think_start.start()] + text[m.start():]
                break
        return text.strip()

    @staticmethod
    def normalize_text(text):
        if not isinstance(text, str):
            text = str(text)
        # Lowercase, split punctuation-delimited values, and collapse whitespace.
        # Replacing punctuation with spaces avoids merging comma-separated IDs.
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())

    # ------------------------------------------------------------------
    # Extraction helpers (used by exact_match)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_action_str(text):
        """Extract 'action_name|action_input_json' from ToolBench format."""
        m = re.search(r'Action:\s*([^\n]+)', text)
        if not m:
            return None
        name = m.group(1).strip()
        args_norm = ""
        inp_m = re.search(r'Action Input:\s*', text)
        if inp_m:
            rest = text[inp_m.end():]
            try:
                obj, _ = json.JSONDecoder().raw_decode(rest.strip())
                args_norm = json.dumps(obj, sort_keys=True)
            except Exception:
                # fallback: take until next newline
                args_norm = rest.split('\n')[0].strip()
        return name.lower() + "|" + args_norm

    @staticmethod
    def _extract_json_from_codeblock(text):
        """Extract JSON content from markdown code blocks (complete or truncated)."""
        # Match complete ```json ... ``` or ``` ... ```
        m = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Handle truncated code block (no closing ```) - common with max_tokens cutoff
        m = re.search(r'```(?:json)?\s*\n?(.*)', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return None

    @staticmethod
    def _normalize_json_calls(text):
        """Extract & normalize a JSON function-call array or single object (ComplexFunc)."""
        def _try_parse(t):
            # Try JSON array first - complete array
            m = re.search(r'\[\s*\{', t)
            if m:
                # Use raw_decode from the [ to handle both complete and truncated
                try:
                    parsed, _ = json.JSONDecoder().raw_decode(t[m.start():])
                    if (isinstance(parsed, list) and
                            all(isinstance(i, dict) and "name" in i for i in parsed)):
                        return parsed
                except Exception:
                    pass
                # Truncated array: extract individual complete objects from partial array
                partial_text = t[m.start() + 1:]  # skip the [
                objects = []
                decoder = json.JSONDecoder()
                pos = 0
                while pos < len(partial_text):
                    # Skip whitespace and commas
                    while pos < len(partial_text) and partial_text[pos] in ' \t\n\r,':
                        pos += 1
                    if pos >= len(partial_text) or partial_text[pos] == ']':
                        break
                    if partial_text[pos] == '{':
                        try:
                            obj, end_idx = decoder.raw_decode(partial_text[pos:])
                            if isinstance(obj, dict) and "name" in obj:
                                objects.append(obj)
                            pos += end_idx
                        except json.JSONDecodeError:
                            break  # truncated object, stop here
                    else:
                        break
                if objects:
                    return objects

            # Try single JSON object {"name": ...}
            m = re.search(r'\{\s*"name"\s*:', t)
            if m:
                try:
                    obj, _ = json.JSONDecoder().raw_decode(t[m.start():])
                    if isinstance(obj, dict) and "name" in obj:
                        return [obj]
                except Exception:
                    pass
            return None

        # First try extracting from code blocks
        codeblock = Metrics._extract_json_from_codeblock(text)
        parsed = None
        if codeblock:
            parsed = _try_parse(codeblock)
        if parsed is None:
            parsed = _try_parse(text)
        if parsed is None:
            return None

        try:
            normalized = []
            for item in parsed:
                args = item.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                normalized.append((
                    item["name"].lower(),
                    json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args)
                ))
            return sorted(normalized)
        except Exception:
            return None

    @staticmethod
    def _extract_final_answer(text):
        """Extract the final answer from a thinking-model output."""
        if not isinstance(text, str):
            return ""
        # 1. After </think> tag
        idx = text.rfind('</think>')
        if idx != -1:
            text = text[idx + len('</think>'):]

        # 2. Look for explicit answer markers
        for pattern in [
            r'(?i)(?:^|\n)\s*(?:ANSWER|Final Answer|Result|Output)\s*[:=]\s*(.+)',
            r'(?i)(?:^|\n)\s*(?:Therefore|Thus|Hence|So),?\s+(?:the answer is|it is)\s*[:.]?\s*(.+)',
            r'(?i)(?:^|\n)\s*The\s+\w+\s+is\s*[:.]?\s*["\']?([^"\'\n]+)["\']?\s*$',
        ]:
            m = re.search(pattern, text)
            if m:
                answer = m.group(1).strip().strip('"\' .')
                if answer:
                    return answer

        # 3. Take last non-empty line(s) — handles multi-line short answers
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if lines:
            return lines[-1]
        return ""

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_core_answer(text):
        """Try multiple strategies to extract only the answer portion from verbose output."""
        if not isinstance(text, str):
            return str(text)

        # Strategy 1: Explicit answer markers
        for pattern in [
            r'(?i)(?:^|\n)\s*(?:ANSWER|Final Answer|Result|Output)\s*[:=]\s*(.+)',
            r'(?i)(?:^|\n)\s*(?:Therefore|Thus|Hence|So),?\s+(?:the answer is|it is)\s*[:.]?\s*(.+)',
        ]:
            m = re.search(pattern, text)
            if m:
                answer = m.group(1).strip().strip('"\' .')
                if answer:
                    return answer

        # Strategy 2: After </think> tag
        idx = text.rfind('</think>')
        if idx != -1:
            after = text[idx + len('</think>'):].strip()
            if after:
                return after

        # Strategy 3: Look for quoted values near answer-indicating phrases
        #  e.g. 'is "DEFA14A"' or 'is "2023-03-15"'
        m = re.search(r'(?:is|=|:)\s*"([^"]+)"\s*[.,;\n]', text)
        if m:
            return m.group(1).strip()

        # Strategy 4: Look for unquoted short values after "is"/"is:"
        #  e.g. 'formType is: DEFA14A' or 'the answer is 42'
        m = re.search(r'(?:is|is:)\s+([A-Za-z0-9][\w\-/., ]{0,50})\s*[.\n*]', text)
        if m:
            candidate = m.group(1).strip().rstrip('.')
            # Only return if it's short enough to be an answer (not a full sentence)
            if len(candidate.split()) <= 8:
                return candidate

        # Strategy 5: Last non-empty line (for verbose outputs)
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if lines:
            return lines[-1]
        return text

    @staticmethod
    def exact_match(prediction, truth):
        # Strip <think> blocks before comparison
        prediction = Metrics._strip_thinking(prediction)
        truth = Metrics._strip_thinking(truth)

        pred_norm = Metrics.normalize_text(prediction)
        truth_norm = Metrics.normalize_text(truth)

        # 1. Raw normalised match
        if pred_norm == truth_norm:
            return 1

        # 2. ToolBench action extraction
        pred_act = Metrics._extract_action_str(prediction)
        truth_act = Metrics._extract_action_str(truth)
        if pred_act and truth_act and pred_act == truth_act:
            return 1

        # 3. ComplexFunc JSON call extraction (handles both [{...}] and {...})
        pred_calls = Metrics._normalize_json_calls(prediction)
        truth_calls = Metrics._normalize_json_calls(truth)
        if pred_calls is not None and truth_calls is not None and pred_calls == truth_calls:
            return 1

        return 0

    @staticmethod
    def contains(prediction, truth):
        prediction = Metrics._strip_thinking(prediction)
        pred_norm = Metrics.normalize_text(prediction)
        truth_norm = Metrics.normalize_text(truth)
        return int(truth_norm in pred_norm)

    @staticmethod
    def f1_score(prediction, truth):
        prediction = Metrics._strip_thinking(prediction)
        if Metrics.exact_match(prediction, truth):
            return 1.0
        # For short ground truths (likely ToolJSON), use extracted answer
        truth_tokens = Metrics.normalize_text(truth).split()
        if len(truth_tokens) <= 10:
            extracted = Metrics._extract_core_answer(prediction)
            if extracted and extracted != prediction:
                prediction = extracted
        pred_tokens = Metrics.normalize_text(prediction).split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
            
        common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common_tokens.values())
        
        if num_same == 0:
            return 0.0
            
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def bleu_score(prediction, truth):
        prediction = Metrics._strip_thinking(prediction)
        truth_tokens_flat = Metrics.normalize_text(truth).split()
        if len(truth_tokens_flat) <= 10:
            extracted = Metrics._extract_core_answer(prediction)
            if extracted and extracted != prediction:
                prediction = extracted
        pred_tokens = Metrics.normalize_text(prediction).split()
        truth_tokens = [truth_tokens_flat]
        smoothie = SmoothingFunction().method4
        return sentence_bleu(truth_tokens, pred_tokens, smoothing_function=smoothie)

    @staticmethod
    def rouge_score(prediction, truth):
        prediction = Metrics._strip_thinking(prediction)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(truth, prediction)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    @staticmethod
    def action_match(prediction, truth):
        """
        Trích xuất tên Action và Arguments (nếu có) và so sánh.
        Hỗ trợ cả format của ToolBench (Action: ... Action Input: ...) và ComplexFuncBench (JSON array).
        """
        prediction = Metrics._strip_thinking(prediction)

        def extract_actions(text):
            if not isinstance(text, str):
                return []

            # First try extracting from code blocks
            codeblock = Metrics._extract_json_from_codeblock(text)
            
            # Try on codeblock first, then full text
            for search_text in ([codeblock, text] if codeblock else [text]):
                # Try JSON array (complete or truncated)
                m = re.search(r'\[\s*\{', search_text)
                if m:
                    # Complete array
                    try:
                        parsed, _ = json.JSONDecoder().raw_decode(search_text[m.start():])
                        if isinstance(parsed, list) and all(isinstance(i, dict) and "name" in i for i in parsed):
                            return [{"name": i["name"], "arguments": i.get("arguments", {})} for i in parsed]
                    except Exception:
                        pass
                    # Truncated array: extract individual complete objects
                    partial = search_text[m.start() + 1:]
                    objects = []
                    decoder = json.JSONDecoder()
                    pos = 0
                    while pos < len(partial):
                        while pos < len(partial) and partial[pos] in ' \t\n\r,':
                            pos += 1
                        if pos >= len(partial) or partial[pos] == ']':
                            break
                        if partial[pos] == '{':
                            try:
                                obj, end_idx = decoder.raw_decode(partial[pos:])
                                if isinstance(obj, dict) and "name" in obj:
                                    objects.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
                                pos += end_idx
                            except json.JSONDecodeError:
                                break
                        else:
                            break
                    if objects:
                        return objects

                # Try single JSON object {"name": ...}
                m = re.search(r'\{\s*"name"\s*:', search_text)
                if m:
                    try:
                        obj, _ = json.JSONDecoder().raw_decode(search_text[m.start():])
                        if isinstance(obj, dict) and "name" in obj:
                            return [{"name": obj["name"], "arguments": obj.get("arguments", {})}]
                    except Exception:
                        pass

            # Fallback: ToolBench format (Action: ... Action Input: ...)
            actions = []
            action_matches = list(re.finditer(r'Action:\s*([^\n]+)', text))
            input_matches = list(re.finditer(r'Action Input:\s*(\{.*?\})', text, re.DOTALL))

            for i, a_match in enumerate(action_matches):
                name = a_match.group(1).strip()
                args = {}
                if i < len(input_matches):
                    try:
                        args = json.loads(input_matches[i].group(1))
                    except json.JSONDecodeError:
                        args = input_matches[i].group(1).strip()
                actions.append({"name": name, "arguments": args})

            return actions

        pred_actions = extract_actions(prediction)
        truth_actions = extract_actions(truth)
        
        if not pred_actions or not truth_actions:
            return 0

        # Nếu là Toolbench (thường chỉ có 1 ground truth action ở bước hiện tại)
        if len(truth_actions) == 1:
            if pred_actions[0]["name"].lower() == truth_actions[0]["name"].lower():
                return 1
            return 0
        
        # Nếu là ComplexFuncBench (có thể có nhiều action gọi song song)
        # Check xem tất cả các name trong truth có xuất hiện trong pred không (bỏ qua args)
        pred_names = [p["name"].lower() for p in pred_actions]
        truth_names = [t["name"].lower() for t in truth_actions]
        
        if sorted(pred_names) == sorted(truth_names):
            return 1
            
        return 0

    @staticmethod
    def llm_as_judge(prediction, truth, question, judge_client):
        """
        Sử dụng LLM để đánh giá xem câu trả lời của mô hình có giải quyết được vấn đề
        hoặc có chứa đúng thông tin như đáp án (truth) hay không.
        """
        prompt = f"""
You are an impartial judge evaluating an AI model's response.
Question/Task: {question}
Ground Truth: {truth}
Model Prediction: {prediction}

Does the model prediction correctly answer the question or contain the essential information from the ground truth? 
Answer with ONLY a valid JSON object containing:
- "reasoning": A short explanation of your judgment.
- "score": 1 if correct, 0 if incorrect.
"""
        try:
            response = judge_client.generate_json(prompt)
            if not response:
                return 0
            # Remove markdown formatting if present
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            result = json.loads(response)
            return result.get("score", 0)
        except Exception as e:
            raise ValueError(f"LLM Judge Error: {e}")
