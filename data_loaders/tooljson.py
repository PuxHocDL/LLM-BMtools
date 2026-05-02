import json
import os
from .base_loader import BaseLoader
from core.enhancers import JSONPruner

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
        api_query = sample.get("api_query", "")
        api_response_path = sample.get("api_response_path", "")

        api_res_full_path = os.path.join(self.base_repo_dir, "generate_qa_pairs", "data", "api_responses", os.path.basename(api_response_path))
        api_res_full_path = api_res_full_path.replace("?", "_").replace(":", "_").replace('"', "_")

        json_content = ""
        try:
            with open(api_res_full_path, 'r', encoding='utf-8') as f:
                raw_json = f.read()
                json_content = JSONPruner.prune(raw_json, question, top_k=15, api_query=api_query)
        except Exception as e:
            json_content = f"<Error loading JSON: {e}>"

        prompt = "Analyze the following JSON output to answer the user's question.\n"
        if api_query:
            prompt += f"API Query Context: {api_query}\n"
        prompt += f"JSON Output:\n{json_content}\n\n"
        prompt += f"Question:\n{question}\n"
        prompt += "Answer directly based on the JSON content."
        return [{"role": "user", "content": prompt}]

    def get_ground_truth(self, sample):
        return sample.get("gold_answer", "")

    def get_question(self, sample):
        return sample.get("question", "Solve the task based on the prompt.")
