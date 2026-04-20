import json
import os
from .base_loader import BaseLoader

class ToolJSONLoader(BaseLoader):
    stop_sequences = []  # No stop sequences for QA-style responses
    enable_thinking = True  # ToolJSON benefits from thinking (answer after </think>)
    
    def __init__(self, data_path, base_repo_dir="data/toolJSONprocessing", agent_name=None):
        self.base_repo_dir = base_repo_dir
        super().__init__(data_path, agent_name=agent_name)

    def load_data(self):
        # data_path is expected to be a directory containing qa_pairs JSON files
        data = []
        if os.path.isdir(self.data_path):
            for filename in os.listdir(self.data_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.data_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            data.extend(file_data)
        else:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        return data

    def format_prompt(self, sample):
        question = sample.get("question", "")
        api_response_path = sample.get("api_response_path", "")
        
        # Fix path for Windows if necessary
        # Bỏ đi tiền tố ../api_responses/ hoặc tương tự
        # Chúng ta giả định api_response_path có thể được đọc từ base_repo_dir
        api_res_full_path = os.path.join(self.base_repo_dir, "generate_qa_pairs", "data", "api_responses", os.path.basename(api_response_path))
        # Handle the ? character which was replaced with _ in our script
        api_res_full_path = api_res_full_path.replace("?", "_").replace(":", "_").replace('"', "_")
        
        json_content = ""
        try:
            with open(api_res_full_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
        except Exception as e:
            json_content = f"<Error loading JSON: {e}>"

        prompt = "Analyze the following JSON output to answer the user's question.\n"
        prompt += f"JSON Output:\n{json_content}\n\n"
        prompt += f"Question:\n{question}\n"
        prompt += "Answer directly based on the JSON content."
        return [{"role": "user", "content": prompt}]

    def get_ground_truth(self, sample):
        return sample.get("gold_answer", "")

    def get_question(self, sample):
        return sample.get("question", "Solve the task based on the prompt.")
