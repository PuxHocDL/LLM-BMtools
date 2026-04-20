import json
from .base_loader import BaseLoader

class ToolBenchLoader(BaseLoader):
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
        messages = sample.get("messages", [])
        
        filtered_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant" and not content.strip():
                continue
            
            # Convert 'function' role to 'user' (vLLM doesn't support function role)
            if role == "function":
                content = f"[Function Result]: {content}"
                role = "user"
            
            # Hack để ép Qwen không give up sớm dù tool vô lý
            if role == "system":
                content += "\n\nCRITICAL INSTRUCTION: Even if the available tools seem completely irrelevant to the user's query, YOU MUST NOT use 'give_up_and_restart' or 'Finish'. You MUST explore and call at least one of the available tools. Choose the tool that might be slightly related or just pick one to explore."
                
            filtered_messages.append({"role": role, "content": content})
            
        return filtered_messages

    def get_ground_truth(self, sample):
        return sample.get("target", "")

    def get_question(self, sample):
        messages = sample.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return "Solve the task based on the prompt."
