import json
import os
import re
from data_loaders.toolbench import ToolBenchLoader
from data_loaders.tooljson import ToolJSONLoader
from core.enhancers import JSONPruner, SemanticToolRetriever

class EnhancedToolBenchLoader(ToolBenchLoader):
    """
    Kế thừa ToolBenchLoader nhưng sử dụng SemanticToolRetriever 
    để lọc bớt các tool không cần thiết trong system prompt.
    """
    def format_prompt(self, sample):
        messages = super().format_prompt(sample)
        
        # Lấy câu hỏi của user để làm query cho Retriever
        user_query = self.get_question(sample)
        
        # Cập nhật system message
        for msg in messages:
            if msg.get("role") == "system":
                content = msg["content"]
                # Tìm chuỗi JSON chứa danh sách API (greedy match)
                match = re.search(r'Specifically, you have access to the following APIs: (\[.*\])', content, re.DOTALL)
                if match:
                    try:
                        tools_json_str = match.group(1)
                        tools_list = json.loads(tools_json_str)
                        
                        # Sử dụng SemanticToolRetriever (giữ lại top 2-3 tools)
                        # Plus Finish tool which is strictly required for ToolBench!
                        finish_tool = {"name": "Finish", "description": "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only parts that will be shown to the user are the final answer, so it should contain sufficient information.", "parameters": {"type": "object", "properties": {"return_type": {"type": "string", "enum": ["give_answer", "give_up_and_restart"]}, "final_answer": {"type": "string", "description": "The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\""}}, "required": ["return_type"]}}
                        
                        filtered_tools = SemanticToolRetriever.retrieve(user_query, tools_list, top_k=3)
                        
                        # Add Finish tool if not present
                        if not any(t.get('name') == 'Finish' for t in filtered_tools):
                            filtered_tools.append(finish_tool)
                            
                        # Replace the old JSON array with the new shortened one
                        new_tools_str = json.dumps(filtered_tools)
                        new_content = content[:match.start(1)] + new_tools_str + content[match.end(1):]
                        msg["content"] = new_content
                    except Exception as e:
                        print(f"Error parsing tools in ToolBench: {e}")
                        pass
        return messages


class EnhancedToolJSONLoader(ToolJSONLoader):
    """
    Kế thừa ToolJSONLoader nhưng sử dụng JSONPruner để tránh lỗi
    Context Length Exceeded bằng cách rút gọn file JSON trước khi feed cho LLM.
    """
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

        prompt = "You are given processed data from an API response. Answer the question using ONLY the data provided.\n"
        prompt += "IMPORTANT: Give a short, direct answer. If it's a number, output just the number. If it's a name/type, output just that value.\n\n"
        if api_query:
            prompt += f"API Query Context: {api_query}\n"
        prompt += f"Data:\n{json_content}\n\n"
        prompt += f"Question: {question}\n"
        prompt += "Answer:"
        return [{"role": "user", "content": prompt}]
