import json
from .base_loader import BaseLoader

class ComplexFuncLoader(BaseLoader):
    def load_data(self):
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data

    def format_prompt(self, sample):
        conversations = sample.get("conversations", [])
        functions = sample.get("functions", [])
        
        system_content = "You are a helpful AI assistant capable of multi-step reasoning and tool calling.\n"
        if functions:
            system_content += "Available Functions:\n" + json.dumps(functions, indent=2) + "\n\n"
        system_content += "Please provide your thought process and the exact tool calls needed.\nIMPORTANT: Your final tool calls must be inside a valid JSON array of objects with 'name' and 'arguments' fields (e.g. [{\"name\": \"func\", \"arguments\": {\"arg1\": \"val\"}}])."
        
        messages = [{"role": "system", "content": system_content}]
        
        for i, msg in enumerate(conversations):
            role = "user" if msg.get("from", msg.get("role")) in ["user", "human"] else "assistant"
            content = msg.get("value", msg.get("content", ""))
            
            # Dừng lại ở lượt trả lời ĐẦU TIÊN của assistant để đánh giá step 1
            if role == "assistant":
                break
                
            messages.append({"role": role, "content": content})
            
        return messages

    def get_ground_truth(self, sample):
        # Lấy lượt gọi tool ĐẦU TIÊN của assistant
        conversations = sample.get("conversations", [])
        for msg in conversations:
            role = "assistant" if msg.get("role") == "assistant" or msg.get("from") == "gpt" else "user"
            if role == "assistant":
                if "function_call" in msg:
                    return json.dumps(msg["function_call"])
                return msg.get("value", msg.get("content", ""))
        return ""

    def get_question(self, sample):
        conversations = sample.get("conversations", [])
        for msg in reversed(conversations):
            if msg.get("role") in ["user", "human"] or msg.get("from") in ["user", "human"]:
                return msg.get("value", msg.get("content", ""))
        return "Solve the task based on the prompt."
