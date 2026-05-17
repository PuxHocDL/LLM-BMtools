import re
from collections import Counter
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

class Metrics:
    @staticmethod
    def _strip_thinking(text):
        """Remove <think>...</think> blocks from thinking-model output."""
        if not isinstance(text, str):
            return str(text)
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    @staticmethod
    def normalize_text(text):
        if not isinstance(text, str):
            text = str(text)
        # Lowercase, remove punctuation and extra whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
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
    def _normalize_json_calls(text):
        """Extract & normalize a JSON function-call array (ComplexFunc)."""
        try:
            m = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
            if not m:
                return None
            parsed = json.loads(m.group(0))
            if not (isinstance(parsed, list) and
                    all(isinstance(i, dict) and "name" in i for i in parsed)):
                return None
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
        # Strip content after </think>
        idx = text.rfind('</think>')
        if idx != -1:
            text = text[idx + len('</think>'):]
        # Take last non-empty line(s) — handles multi-line short answers
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if lines:
            return lines[-1]
        return ""

    # ------------------------------------------------------------------

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

        # 3. ComplexFunc JSON call extraction
        pred_calls = Metrics._normalize_json_calls(prediction)
        truth_calls = Metrics._normalize_json_calls(truth)
        if pred_calls is not None and truth_calls is not None and pred_calls == truth_calls:
            return 1

        # 4. Final-answer extraction (last non-empty line)
        pred_answer = Metrics._extract_final_answer(prediction)
        if pred_answer:
            if Metrics.normalize_text(pred_answer) == truth_norm:
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
        pred_tokens = Metrics.normalize_text(prediction).split()
        truth_tokens = Metrics.normalize_text(truth).split()

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
        pred_tokens = Metrics.normalize_text(prediction).split()
        truth_tokens = [Metrics.normalize_text(truth).split()]
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

            # Try to parse as JSON array (ComplexFuncBench format)
            try:
                # Tìm mảng JSON đầu tiên trong text
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, list) and all(isinstance(i, dict) and "name" in i for i in parsed):
                        return [{"name": i["name"], "arguments": i.get("arguments", {})} for i in parsed]
            except Exception:
                pass

            # Nếu không phải JSON array, thử parse dạng ToolBench
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
