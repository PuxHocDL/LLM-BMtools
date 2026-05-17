"""
5 Enhancement Strategies for the Evaluation Pipeline.

Strategy 1: Prompt Rewrite — Better output format instructions
Strategy 2: Tool Compression — Keep ALL tools but compress descriptions
Strategy 3: Chain-of-Thought — Add reasoning guidance
Strategy 4: Two-Stage LLM — Use LLM to preprocess, then answer
Strategy 5: Few-Shot — Add in-context examples
"""

import json
import os
import re
from data_loaders.toolbench import ToolBenchLoader
from data_loaders.complexfunc import ComplexFuncLoader
from data_loaders.tooljson import ToolJSONLoader
from core.enhancers import JSONPruner, SemanticToolRetriever


# =============================================================================
# Helper: extract tools JSON from ToolBench system message
# =============================================================================
def _extract_tools_from_system(content):
    """Extract the tools JSON array from a ToolBench system message."""
    match = re.search(r'Specifically, you have access to the following APIs: (\[.*\])', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1)), match.start(1), match.end(1)
        except Exception:
            pass
    return None, None, None


def _replace_tools_in_content(content, start, end, new_tools):
    """Replace the tools JSON in the system message content."""
    return content[:start] + json.dumps(new_tools) + content[end:]


def _load_json_for_tooljson(sample, base_repo_dir="data/toolJSONprocessing"):
    """Load and return raw JSON content for a ToolJSON sample."""
    api_response_path = sample.get("api_response_path", "")
    api_res_full_path = os.path.join(base_repo_dir, "generate_qa_pairs", "data", "api_responses", os.path.basename(api_response_path))
    api_res_full_path = api_res_full_path.replace("?", "_").replace(":", "_").replace('"', "_")
    try:
        with open(api_res_full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"<Error loading JSON: {e}>"


def _sanitize_role(role, content):
    """Convert unsupported message roles (e.g. 'function') to 'user'."""
    if role == "function":
        return "user", f"[Function Result]: {content}"
    return role, content


# =============================================================================
# STRATEGY 1: Prompt Rewrite
# =============================================================================

class S1_ToolBench(ToolBenchLoader):
    """Prompt Rewrite: strict output format, remove CRITICAL hack."""
    def format_prompt(self, sample):
        messages = sample.get("messages", [])
        filtered_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            role, content = _sanitize_role(role, content)
            if role == "assistant" and not content.strip():
                continue
            if role == "system":
                # Replace the vague instructions with strict format
                content += "\n\nOUTPUT FORMAT: You MUST respond in EXACTLY this format, nothing else:\nThought: <your reasoning about which tool to use and why>\nAction: <exact tool name from the list above>\nAction Input: <valid JSON object with the required parameters>"
            filtered_messages.append({"role": role, "content": content})
        return filtered_messages


class S1_ComplexFunc(ComplexFuncLoader):
    """Prompt Rewrite: structured step-by-step + strict JSON output."""
    def format_prompt(self, sample):
        conversations = sample.get("conversations", [])
        functions = sample.get("functions", [])

        system_content = "You are a function-calling AI assistant.\n"
        if functions:
            system_content += "Available Functions:\n" + json.dumps(functions, indent=2) + "\n\n"
        system_content += (
            "Instructions:\n"
            "1. Read the user request carefully.\n"
            "2. Identify which function(s) to call and with what arguments.\n"
            "3. Output ONLY a JSON array of function calls. No explanation.\n"
            "Format: [{\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}]\n"
        )

        messages = [{"role": "system", "content": system_content}]
        for msg in conversations:
            role = "user" if msg.get("from", msg.get("role")) in ["user", "human"] else "assistant"
            content = msg.get("value", msg.get("content", ""))
            if role == "assistant":
                break
            messages.append({"role": role, "content": content})
        return messages


class S1_ToolJSON(ToolJSONLoader):
    """Prompt Rewrite: direct answer instruction with pruned JSON."""
    stop_sequences = []

    def format_prompt(self, sample):
        question = sample.get("question", "")
        api_query = sample.get("api_query", "")
        raw_json = _load_json_for_tooljson(sample, self.base_repo_dir)
        json_content = JSONPruner.prune(raw_json, question, top_k=15, api_query=api_query)

        prompt = (
            "You are a data extraction assistant. Given API response data and a question, "
            "output ONLY the answer value. No explanation, no sentences.\n"
            "- If the answer is a number, output just the number.\n"
            "- If the answer is a name/type/date, output just that value.\n"
            "- If multiple values, output comma-separated.\n\n"
        )
        if api_query:
            prompt += f"API Query Context: {api_query}\n"
        prompt += (
            f"Data:\n{json_content}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return [{"role": "user", "content": prompt}]


# =============================================================================
# STRATEGY 2: Tool Compression
# =============================================================================

class ToolCompressor:
    """Compress tool descriptions: keep name, 1-line desc, param names only."""
    @staticmethod
    def compress_tool(tool):
        name = tool.get("name", "")
        desc = tool.get("description", "")
        # Truncate description to first sentence
        first_sentence = desc.split('.')[0].strip() + '.' if desc else ""
        if len(first_sentence) > 120:
            first_sentence = first_sentence[:117] + "..."

        # Extract just parameter names
        params = tool.get("parameters", {})
        param_names = []
        if isinstance(params, dict):
            props = params.get("properties", {})
            required = params.get("required", [])
            for pname in props:
                suffix = " (required)" if pname in required else ""
                param_names.append(f"{pname}{suffix}")

        compressed = {"name": name, "description": first_sentence}
        if param_names:
            compressed["params"] = ", ".join(param_names)
        return compressed

    @classmethod
    def compress_tools(cls, tools_list):
        return [cls.compress_tool(t) for t in tools_list]


class S2_ToolBench(ToolBenchLoader):
    """Tool Compression: keep ALL tools but compress descriptions."""
    def format_prompt(self, sample):
        messages = sample.get("messages", [])
        filtered_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            role, content = _sanitize_role(role, content)
            if role == "assistant" and not content.strip():
                continue
            if role == "system":
                tools, start, end = _extract_tools_from_system(content)
                if tools and start is not None:
                    compressed = ToolCompressor.compress_tools(tools)
                    content = _replace_tools_in_content(content, start, end, compressed)
                content += "\n\nRespond with:\nThought: <reasoning>\nAction: <tool_name>\nAction Input: <json>"
            filtered_messages.append({"role": role, "content": content})
        return filtered_messages


class S2_ComplexFunc(ComplexFuncLoader):
    """Tool Compression: compress function signatures."""
    def format_prompt(self, sample):
        conversations = sample.get("conversations", [])
        functions = sample.get("functions", [])

        compressed_funcs = []
        for func in functions:
            cf = {"name": func.get("name", "")}
            desc = func.get("description", "")
            cf["description"] = (desc.split('.')[0].strip() + '.') if desc else ""
            params = func.get("parameters", {})
            if isinstance(params, dict):
                props = params.get("properties", {})
                required = params.get("required", [])
                cf["parameters"] = {
                    pname: {"type": pinfo.get("type", "string")}
                    for pname, pinfo in props.items()
                }
                if required:
                    cf["required"] = required
            compressed_funcs.append(cf)

        system_content = "You are a helpful AI assistant capable of tool calling.\n"
        if compressed_funcs:
            system_content += "Available Functions:\n" + json.dumps(compressed_funcs, indent=2) + "\n\n"
        system_content += "Output a JSON array of function calls: [{\"name\": \"func\", \"arguments\": {\"arg\": \"val\"}}]"

        messages = [{"role": "system", "content": system_content}]
        for msg in conversations:
            role = "user" if msg.get("from", msg.get("role")) in ["user", "human"] else "assistant"
            if role == "assistant":
                break
            messages.append({"role": role, "content": msg.get("value", msg.get("content", ""))})
        return messages


class S2_ToolJSON(ToolJSONLoader):
    """Tool Compression: same as S1 for ToolJSON (pruning is the compression)."""
    stop_sequences = []

    def format_prompt(self, sample):
        question = sample.get("question", "")
        api_query = sample.get("api_query", "")
        raw_json = _load_json_for_tooljson(sample, self.base_repo_dir)
        json_content = JSONPruner.prune(raw_json, question, top_k=15, api_query=api_query)

        prompt = ""
        if api_query:
            prompt += f"API Query Context: {api_query}\n"
        prompt += (
            f"Data:\n{json_content}\n\n"
            f"Question: {question}\n"
            "Answer with ONLY the value, nothing else:"
        )
        return [{"role": "user", "content": prompt}]


# =============================================================================
# STRATEGY 3: Chain-of-Thought
# =============================================================================

class S3_ToolBench(ToolBenchLoader):
    """CoT: guide model to reason before selecting tool."""
    def format_prompt(self, sample):
        messages = sample.get("messages", [])
        filtered_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            role, content = _sanitize_role(role, content)
            if role == "assistant" and not content.strip():
                continue
            if role == "system":
                content += (
                    "\n\nBefore selecting a tool, follow these reasoning steps:"
                    "\n1. What does the user need? List the key requirements."
                    "\n2. Which available tools could address each requirement? Match tool descriptions to needs."
                    "\n3. Which single tool is the BEST first step? Pick that one."
                    "\n\nThen respond with:"
                    "\nThought: <your step-by-step reasoning>"
                    "\nAction: <the chosen tool name>"
                    "\nAction Input: <JSON arguments>"
                )
            filtered_messages.append({"role": role, "content": content})
        return filtered_messages


class S3_ComplexFunc(ComplexFuncLoader):
    """CoT: decompose request into sub-tasks then map to functions."""
    def format_prompt(self, sample):
        conversations = sample.get("conversations", [])
        functions = sample.get("functions", [])

        system_content = "You are a helpful AI assistant capable of multi-step reasoning and tool calling.\n"
        if functions:
            system_content += "Available Functions:\n" + json.dumps(functions, indent=2) + "\n\n"
        system_content += (
            "Follow these steps:\n"
            "1. DECOMPOSE: Break the user request into atomic sub-tasks.\n"
            "2. MAP: For each sub-task, identify which function to call and what arguments to use.\n"
            "3. OUTPUT: Provide your reasoning, then output the function calls as a JSON array.\n"
            "Format: [{\"name\": \"func\", \"arguments\": {\"arg\": \"val\"}}]\n"
        )

        messages = [{"role": "system", "content": system_content}]
        for msg in conversations:
            role = "user" if msg.get("from", msg.get("role")) in ["user", "human"] else "assistant"
            if role == "assistant":
                break
            messages.append({"role": role, "content": msg.get("value", msg.get("content", ""))})
        return messages


class S3_ToolJSON(ToolJSONLoader):
    """CoT: guide model to locate data first, then extract answer."""
    stop_sequences = []

    def format_prompt(self, sample):
        question = sample.get("question", "")
        api_query = sample.get("api_query", "")
        raw_json = _load_json_for_tooljson(sample, self.base_repo_dir)
        json_content = JSONPruner.prune(raw_json, question, top_k=15, api_query=api_query)

        prompt = ""
        if api_query:
            prompt += f"API Query Context: {api_query}\n"
        prompt += (
            f"Data:\n{json_content}\n\n"
            f"Question: {question}\n\n"
            "Think step by step:\n"
            "1. Which field(s) in the data are relevant to this question?\n"
            "2. Which record(s) match the conditions in the question?\n"
            "3. What is the exact answer value?\n\n"
            "After reasoning, write your final answer on the last line after 'ANSWER: '."
        )
        return [{"role": "user", "content": prompt}]


# =============================================================================
# STRATEGY 4: Two-Stage LLM
# =============================================================================

class S4_ToolBench(ToolBenchLoader):
    """Two-Stage: LLM selects relevant tools first, then acts with those tools."""
    def __init__(self, data_path, agent_name=None):
        super().__init__(data_path, agent_name=agent_name)
        from core.llm_client import LLMClient
        self._preprocessor = LLMClient(agent_name=agent_name or "qwen")

    def format_prompt(self, sample):
        messages = sample.get("messages", [])
        user_query = self.get_question(sample)

        # Find system message and extract tools
        system_content = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
                break

        tools, start, end = _extract_tools_from_system(system_content)
        if not tools or start is None:
            return super().format_prompt(sample)

        # Stage 1: Ask LLM to pick the best tools
        tool_summary = "\n".join(
            f"- {t.get('name','')}: {t.get('description','')[:100]}" for t in tools
        )
        stage1_prompt = (
            f"User query: {user_query}\n\n"
            f"Available tools:\n{tool_summary}\n\n"
            "Which 3 tools are most relevant to this query? "
            "Output ONLY the tool names, one per line, nothing else."
        )
        stage1_result = self._preprocessor.generate(stage1_prompt, stop=[])
        selected_names = set(line.strip().strip('-').strip() for line in stage1_result.strip().split('\n') if line.strip())

        # Filter tools to only selected ones
        filtered_tools = [t for t in tools if t.get('name', '') in selected_names]

        # Always keep Finish tool
        finish_names = {t.get('name') for t in filtered_tools}
        if 'Finish' not in finish_names:
            finish_tool = next((t for t in tools if t.get('name') == 'Finish'), None)
            if finish_tool:
                filtered_tools.append(finish_tool)

        # If LLM didn't select anything useful, fall back to all tools
        if len(filtered_tools) <= 1:
            filtered_tools = tools

        # Build final messages with filtered tools
        filtered_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            role, content = _sanitize_role(role, content)
            if role == "assistant" and not content.strip():
                continue
            if role == "system":
                content = _replace_tools_in_content(content, start, end, filtered_tools)
                content += "\n\nRespond with:\nThought: <reasoning>\nAction: <tool_name>\nAction Input: <json>"
            filtered_messages.append({"role": role, "content": content})
        return filtered_messages


class S4_ComplexFunc(ComplexFuncLoader):
    """Two-Stage: LLM identifies relevant functions first, then calls them."""
    def __init__(self, data_path, agent_name=None):
        super().__init__(data_path, agent_name=agent_name)
        from core.llm_client import LLMClient
        self._preprocessor = LLMClient(agent_name=agent_name or "qwen")

    def format_prompt(self, sample):
        conversations = sample.get("conversations", [])
        functions = sample.get("functions", [])

        if not functions:
            return super().format_prompt(sample)

        # Get user question
        user_query = ""
        for msg in conversations:
            role = "user" if msg.get("from", msg.get("role")) in ["user", "human"] else "assistant"
            if role == "user":
                user_query = msg.get("value", msg.get("content", ""))
            elif role == "assistant":
                break

        # Stage 1: Ask LLM which functions are needed
        func_summary = "\n".join(
            f"- {f.get('name','')}: {f.get('description','')[:100]}" for f in functions
        )
        stage1_prompt = (
            f"User request: {user_query}\n\n"
            f"Available functions:\n{func_summary}\n\n"
            "Which functions are needed? Output ONLY function names, one per line."
        )
        stage1_result = self._preprocessor.generate(stage1_prompt, stop=[])
        selected_names = set(line.strip().strip('-').strip() for line in stage1_result.strip().split('\n') if line.strip())

        # Filter functions
        filtered_funcs = [f for f in functions if f.get('name', '') in selected_names]
        if not filtered_funcs:
            filtered_funcs = functions

        # Stage 2: Build prompt with only relevant functions
        system_content = "You are a function-calling AI assistant.\n"
        system_content += "Available Functions:\n" + json.dumps(filtered_funcs, indent=2) + "\n\n"
        system_content += "Output ONLY a JSON array of function calls: [{\"name\": \"func\", \"arguments\": {\"arg\": \"val\"}}]"

        messages = [{"role": "system", "content": system_content}]
        for msg in conversations:
            role = "user" if msg.get("from", msg.get("role")) in ["user", "human"] else "assistant"
            if role == "assistant":
                break
            messages.append({"role": role, "content": msg.get("value", msg.get("content", ""))})
        return messages


class S4_ToolJSON(ToolJSONLoader):
    """Two-Stage: LLM extracts relevant snippet first, then answers."""
    stop_sequences = []

    def __init__(self, data_path, base_repo_dir="data/toolJSONprocessing", agent_name=None):
        super().__init__(data_path, base_repo_dir, agent_name=agent_name)
        from core.llm_client import LLMClient
        self._preprocessor = LLMClient(agent_name=agent_name or "qwen")

    def format_prompt(self, sample):
        question = sample.get("question", "")
        api_query = sample.get("api_query", "")
        raw_json = _load_json_for_tooljson(sample, self.base_repo_dir)
        pruned = JSONPruner.prune(raw_json, question, top_k=15, api_query=api_query)

        # Stage 1: Ask LLM to extract just the relevant data
        stage1_prompt = ""
        if api_query:
            stage1_prompt += f"API Query Context: {api_query}\n"
        stage1_prompt += (
            f"Data:\n{pruned}\n\n"
            f"Question: {question}\n\n"
            "Extract ONLY the data records or values needed to answer this question. "
            "Output just the relevant data, nothing else."
        )
        extracted = self._preprocessor.generate(stage1_prompt, stop=[])

        # Stage 2: Answer based on extracted data
        prompt = (
            f"Extracted data:\n{extracted}\n\n"
            f"Question: {question}\n"
            "Answer with ONLY the value, nothing else:"
        )
        return [{"role": "user", "content": prompt}]


# =============================================================================
# STRATEGY 5: Few-Shot In-Context Examples
# =============================================================================

# Hardcoded examples for each dataset
_TOOLBENCH_EXAMPLE = {
    "user": "I need to find the weather forecast for Paris next week and also get restaurant recommendations nearby.",
    "response": (
        "Thought: The user wants weather information for Paris. I should use the weather API first to get the forecast.\n"
        "Action: get_weather_forecast\n"
        'Action Input: {"city": "Paris", "days": 7}'
    )
}

_COMPLEXFUNC_EXAMPLE = {
    "user": "Book a flight from New York to London on December 25th for 2 adults in economy class.",
    "response": '[{"name": "search_flights", "arguments": {"origin": "New York", "destination": "London", "date": "2024-12-25", "passengers": 2, "class": "economy"}}]'
}

_TOOLJSON_EXAMPLE = {
    "data": '{"products": [{"id": "123", "name": "Nike Air Max", "price": 129.99, "department": "Men\'s"}]}',
    "question": "What department does product 123 belong to?",
    "answer": "Men's"
}


class S5_ToolBench(ToolBenchLoader):
    """Few-Shot: add 1 example before the actual task."""
    def format_prompt(self, sample):
        messages = sample.get("messages", [])
        filtered_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            role, content = _sanitize_role(role, content)
            if role == "assistant" and not content.strip():
                continue
            if role == "system":
                content += (
                    "\n\n--- EXAMPLE ---"
                    f"\nUser: {_TOOLBENCH_EXAMPLE['user']}"
                    f"\nAssistant: {_TOOLBENCH_EXAMPLE['response']}"
                    "\n--- END EXAMPLE ---"
                    "\n\nNow respond to the actual task in the same format:"
                    "\nThought: <reasoning>"
                    "\nAction: <tool_name>"
                    "\nAction Input: <json>"
                )
            filtered_messages.append({"role": role, "content": content})
        return filtered_messages


class S5_ComplexFunc(ComplexFuncLoader):
    """Few-Shot: add 1 function-calling example."""
    def format_prompt(self, sample):
        conversations = sample.get("conversations", [])
        functions = sample.get("functions", [])

        system_content = "You are a helpful AI assistant capable of tool calling.\n"
        if functions:
            system_content += "Available Functions:\n" + json.dumps(functions, indent=2) + "\n\n"
        system_content += (
            "Output a JSON array of function calls.\n\n"
            "--- EXAMPLE ---\n"
            f"User: {_COMPLEXFUNC_EXAMPLE['user']}\n"
            f"Assistant: {_COMPLEXFUNC_EXAMPLE['response']}\n"
            "--- END EXAMPLE ---\n\n"
            "Now handle the actual request. Output ONLY the JSON array:"
        )

        messages = [{"role": "system", "content": system_content}]
        for msg in conversations:
            role = "user" if msg.get("from", msg.get("role")) in ["user", "human"] else "assistant"
            if role == "assistant":
                break
            messages.append({"role": role, "content": msg.get("value", msg.get("content", ""))})
        return messages


class S5_ToolJSON(ToolJSONLoader):
    """Few-Shot: add 1 QA example before the actual question."""
    stop_sequences = []

    def format_prompt(self, sample):
        question = sample.get("question", "")
        api_query = sample.get("api_query", "")
        raw_json = _load_json_for_tooljson(sample, self.base_repo_dir)
        json_content = JSONPruner.prune(raw_json, question, top_k=15, api_query=api_query)

        prompt = "You are a data extraction assistant.\n\n"
        prompt += (
            "--- EXAMPLE ---\n"
            f"Data: {_TOOLJSON_EXAMPLE['data']}\n"
            f"Question: {_TOOLJSON_EXAMPLE['question']}\n"
            f"Answer: {_TOOLJSON_EXAMPLE['answer']}\n"
            "--- END EXAMPLE ---\n\n"
        )
        if api_query:
            prompt += f"API Query Context: {api_query}\n"
        prompt += (
            f"Data:\n{json_content}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return [{"role": "user", "content": prompt}]


# =============================================================================
# STRATEGY 6: Context-Aware Pipeline
# =============================================================================

class S6_ToolBench(ToolBenchLoader):
    """Context-Aware: S1 prompt rewrite (best performer) + query-scoped context."""
    def format_prompt(self, sample):
        messages = sample.get("messages", [])
        filtered_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            role, content = _sanitize_role(role, content)
            if role == "assistant" and not content.strip():
                continue
            if role == "system":
                content += "\n\nOUTPUT FORMAT: You MUST respond in EXACTLY this format, nothing else:\nThought: <your reasoning about which tool to use and why>\nAction: <exact tool name from the list above>\nAction Input: <valid JSON object with the required parameters>"
            filtered_messages.append({"role": role, "content": content})
        return filtered_messages


class S6_ComplexFunc(ComplexFuncLoader):
    """Context-Aware: S1 prompt rewrite (best performer)."""
    def format_prompt(self, sample):
        conversations = sample.get("conversations", [])
        functions = sample.get("functions", [])

        system_content = "You are a function-calling AI assistant.\n"
        if functions:
            system_content += "Available Functions:\n" + json.dumps(functions, indent=2) + "\n\n"
        system_content += (
            "Instructions:\n"
            "1. Read the user request carefully.\n"
            "2. Identify which function(s) to call and with what arguments.\n"
            "3. Output ONLY a JSON array of function calls. No explanation.\n"
            "Format: [{\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}]\n"
        )

        messages = [{"role": "system", "content": system_content}]
        for msg in conversations:
            role = "user" if msg.get("from", msg.get("role")) in ["user", "human"] else "assistant"
            content = msg.get("value", msg.get("content", ""))
            if role == "assistant":
                break
            messages.append({"role": role, "content": content})
        return messages


class S6_ToolJSON(ToolJSONLoader):
    """Context-Aware: task-type routing + query-scoped pruning."""
    stop_sequences = []

    _TASK_PROMPTS = {
        "EXTRACTIVE": (
            "You are a data extraction assistant.\n"
            "Given the API response data below, find the EXACT value that answers the question.\n"
            "Output ONLY the value itself — no explanation, no full sentences.\n\n"
        ),
        "AGGREGATION": (
            "You are a data analysis assistant.\n"
            "Given the data summary below, count the records that match the question's conditions.\n"
            "Output ONLY the number — no explanation, no units, no sentences.\n\n"
        ),
        "FILTERING": (
            "You are a data extraction assistant.\n"
            "Given the API response data below, find all unique values that match the question's conditions.\n"
            "Output ONLY the values as a comma-separated list, sorted alphabetically.\n"
            "Example format: A, B, C, D\n\n"
        ),
    }

    def format_prompt(self, sample):
        question = sample.get("question", "")
        api_query = sample.get("api_query", "")
        task_type = sample.get("task_type", "EXTRACTIVE")
        raw_json = _load_json_for_tooljson(sample, self.base_repo_dir)
        json_content = JSONPruner.prune(raw_json, question, top_k=15, api_query=api_query)

        prompt = self._TASK_PROMPTS.get(task_type, self._TASK_PROMPTS["EXTRACTIVE"])
        if api_query:
            prompt += f"API Query Context: {api_query}\n"
        prompt += (
            f"Data:\n{json_content}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return [{"role": "user", "content": prompt}]


# =============================================================================
# STRATEGY 7: Self-Debug
# =============================================================================

class S7_SelfDebug_ToolJSON(ToolJSONLoader):
    """Self-Debug: Generates Python code to extract answer and debugs if it fails."""
    stop_sequences = []

    @staticmethod
    def _stringify_answer(value):
        if value is None:
            return ""
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, float):
            try:
                if value == int(value):
                    return str(int(value))
                return str(value)
            except Exception:
                pass
        text = str(value).strip()
        if re.fullmatch(r"[-+]?\d+\.\d{10,}", text):
            try:
                return f"{float(text):.16g}"
            except Exception:
                return text
        return text

    @staticmethod
    def _payload_excerpt(payload, max_chars=12000):
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... (payload excerpt truncated; solve(data) will receive the full payload)"

    @classmethod
    def _shape_summary(cls, payload, max_paths=80):
        rows = []

        def walk(value, path, depth):
            if len(rows) >= max_paths or depth > 5:
                return
            if isinstance(value, dict):
                keys = list(value.keys())
                rows.append(f"{path}: dict keys={keys[:18]}")
                for key in keys[:12]:
                    walk(value[key], f"{path}.{key}", depth + 1)
            elif isinstance(value, list):
                rows.append(f"{path}: list len={len(value)}")
                if value:
                    walk(value[0], f"{path}[0]", depth + 1)
            else:
                rows.append(f"{path}: {type(value).__name__} example={repr(value)[:80]}")

        walk(payload, "data", 0)
        return "\n".join(rows)

    @classmethod
    def _primary_list_paths(cls, payload):
        paths = []

        def walk(value, path):
            if isinstance(value, list):
                paths.append((path, len(value)))
                if value:
                    walk(value[0], f"{path}[0]")
            elif isinstance(value, dict):
                for key, child in value.items():
                    walk(child, f"{path}.{key}")

        walk(payload, "data")
        return [f"{path} (len={length})" for path, length in paths[:12]]

    @classmethod
    def _candidate_value_summary(cls, payload, max_items=80):
        values = []

        def is_candidate_key(key):
            key = str(key).lower()
            generic_keys = {"id", "name", "title", "type", "date", "currency", "department"}
            generic_suffixes = ("_id", "_name", "_title", "_type", "_date", "_currency")
            return key in generic_keys or key.endswith(generic_suffixes)

        def walk(value, path):
            if len(values) >= max_items:
                return
            if isinstance(value, dict):
                for key, child in value.items():
                    child_path = f"{path}.{key}"
                    if is_candidate_key(key) and isinstance(child, (str, int, float)):
                        values.append(f"{child_path} = {child!r}")
                    walk(child, child_path)
            elif isinstance(value, list):
                for idx, item in enumerate(value[:20]):
                    walk(item, f"{path}[{idx}]")

        walk(payload, "data")
        return "\n".join(values[:max_items]) if values else "No candidate text/id values found."

    @staticmethod
    def _extract_python_code(model_output):
        if not isinstance(model_output, str):
            return ""

        fenced = re.search(r"```(?:python)?\s*(.*?)```", model_output, re.DOTALL | re.IGNORECASE)
        if fenced:
            code = fenced.group(1).strip()
        else:
            idx = model_output.find("def solve")
            code = model_output[idx:].strip() if idx != -1 else model_output.strip()

        # Drop trailing prose after the function if the model ignored instructions.
        marker_patterns = [
            r"\n(?:Explanation|Notes?|This function|The function)\b.*$",
            r"\n```.*$",
        ]
        for pattern in marker_patterns:
            code = re.sub(pattern, "", code, flags=re.DOTALL | re.IGNORECASE).strip()
        return S7_SelfDebug_ToolJSON._repair_case_insensitive_literals(code)

    @staticmethod
    def _repair_case_insensitive_literals(code):
        """Fix common model bug: 'Needle' in value.lower() is always false."""
        if not code:
            return code

        def lower_literal(match):
            quote = match.group("quote")
            literal = match.group("literal")
            op = match.group("op")
            rhs = match.group("rhs")
            return f"{quote}{literal.lower()}{quote} {op} {rhs}"

        code = re.sub(
            r"(?P<quote>['\"])(?P<literal>[^'\"]*[A-Z][^'\"]*)(?P=quote)\s+"
            r"(?P<op>in)\s+(?P<rhs>[^\n:]+?\.lower\(\))",
            lower_literal,
            code,
        )

        def lower_rhs_literal(match):
            lhs = match.group("lhs")
            op = match.group("op")
            quote = match.group("quote")
            literal = match.group("literal")
            return f"{lhs} {op} {quote}{literal.lower()}{quote}"

        code = re.sub(
            r"(?P<lhs>[^\n:=]+?\.lower\(\))\s*(?P<op>==|!=)\s*"
            r"(?P<quote>['\"])(?P<literal>[^'\"]*[A-Z][^'\"]*)(?P=quote)",
            lower_rhs_literal,
            code,
        )

        code = re.sub(
            r"(?P<quote>['\"])(?P<literal>[^'\"]*[A-Z][^'\"]*)(?P=quote)\s*"
            r"(?P<op>==|!=)\s*(?P<rhs>[^\n:=]+?\.lower\(\))",
            lower_literal,
            code,
        )
        return code

    def _question_guidance(self, sample):
        question = sample.get("question", "")
        task = sample.get("task", "")
        q = self._norm_text(question)
        lines = []

        if task == "GetRoomCount" or "total number of available rooms of the kind" in q:
            room_name = self._extract_room_name(question, "What is the total number of available rooms of the kind ")
            if room_name:
                lines.extend([
                    f"Parsed room name: {room_name!r}",
                    "Use exact normalized equality against room['name']; do not use substring matching.",
                    "Do not read room['room_name'] for this task; it can omit policy text and fail exact matching.",
                    "Return the matched room's `available` field if present, otherwise `room_count`; do not count matching records.",
                ])

        elif task == "GetRoomArea" or "area in square feet" in q:
            room_name = self._extract_room_name(
                question,
                "What is the area in square feet of ",
                "? Include just the number and not the unit.",
            )
            if room_name:
                lines.extend([
                    f"Parsed room name: {room_name!r}",
                    "Use exact normalized equality against room['name']; do not use substring matching.",
                    "Do not read room['room_name'] for this task; it can omit policy text and fail exact matching.",
                    "Return room['room_surface_in_feet2'] exactly from the matched room; do not round, cast to int, or convert units.",
                ])

        elif task == "GetRoomsWithPriceLessThanAmount" or "gross rate less than" in q:
            amount_match = re.search(r"less than \$([0-9.]+)\s*USD", question)
            if amount_match:
                lines.extend([
                    f"Parsed gross-rate threshold: {amount_match.group(1)}",
                    "For each room in data['available'], compare product_price_breakdown['gross_amount_per_night']['value'] to the threshold.",
                    "Append the exact room['name'] for matching rooms; never use room_name, name_without_policy, or display price strings.",
                ])

        elif task == "GetRoomsWithMealPlan" or q.startswith("get rooms with"):
            mealplan = self._extract_quoted(question)
            if mealplan:
                lines.extend([
                    f"Parsed meal plan: {mealplan!r}",
                    "Compare the parsed meal plan with room['mealplan'] using normalized equality.",
                    "Append the exact room['name'] for matching rooms; never use room_name or name_without_policy.",
                ])

        elif task == "GetLowestCost" or "all inclusive cost" in q:
            lines.append("Compute the minimum of product_price_breakdown['all_inclusive_amount']['value'] across data['available']; return the numeric value exactly.")

        elif task == "GetHighestVAT" or "highest vat" in q:
            lines.append("Inspect product_price_breakdown['items']; find items named 'VAT' and return the maximum item_amount['value'].")

        elif task == "GetInsurancePrice" or "travel insurance plan" in q:
            plan = self._extract_quoted(question)
            if plan:
                lines.append(f"Parsed insurance plan: {plan!r}")
            lines.append("travelInsurance['options'] is a dict, not a list. Never write `for option in options`.")
            lines.append("If options['type'] matches, read options['priceBreakdown']['total'] and return '<currencyCode> <units + nanos/1_000_000_000>'.")
            lines.append("Canonical pattern: options = data['data']['travelInsurance']['options']; total = options['priceBreakdown']['total']; amount = total['units'] + total.get('nanos', 0) / 1_000_000_000.")

        elif task == "GetLuggageAllowance" or "luggage allowance" in q:
            lines.append("If data['data'] has checkedInBaggage, read checkedInBaggage['options'][0]['luggageAllowance']; otherwise use cabinBaggagePerTraveller['luggageAllowance'].")
            lines.append("Return '<luggageType> <maxWeightPerPiece><massUnit>', for example CHECKED_IN 50.7LB.")
            lines.append("Canonical pattern starts with payload = data['data']; do not look for checkedInBaggage at the root.")

        return "\n".join(f"- {line}" for line in lines) if lines else "- No extra parsed guidance."

    def _extract_target_json(self, raw_json_str, api_query):
        """Helper to extract the actual useful payload from the nested structure."""
        try:
            data = json.loads(raw_json_str)
            query_key = api_query.replace("'", '"')
            for domain_val in data.values():
                if isinstance(domain_val, dict):
                    for endpoint_data in domain_val.values():
                        if not isinstance(endpoint_data, dict):
                            continue
                        if query_key in endpoint_data:
                            return endpoint_data[query_key]
                        for key, value in endpoint_data.items():
                            if str(key).strip() == str(query_key).strip():
                                return value
            return data
        except Exception:
            return {}

    @staticmethod
    def _extract_quoted(text):
        match = re.search(r'"([^"]+)"', text or "", flags=re.DOTALL)
        return match.group(1) if match else None

    @staticmethod
    def _extract_room_name(question, prefix, suffix=None):
        pattern = re.escape(prefix) + r"\s*(.+?)"
        if suffix:
            pattern += re.escape(suffix)
        else:
            pattern += r"\?"
        match = re.search(pattern, question or "", flags=re.DOTALL)
        if not match:
            return None
        return re.sub(r"\s+", " ", match.group(1)).strip()

    @staticmethod
    def _norm_text(text):
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    def _find_room_by_question_name(self, api_response, room_name):
        if not room_name:
            return None
        target = self._norm_text(room_name)
        rooms = api_response.get("available", []) if isinstance(api_response, dict) else []
        for room in rooms:
            if self._norm_text(room.get("name")) == target:
                return room
        for room in rooms:
            if self._norm_text(room.get("room_name")) == target:
                return room
        for room in rooms:
            name = self._norm_text(room.get("name"))
            if target in name or name in target:
                return room
        return None

    @staticmethod
    def _uses_substring_name_match(code):
        key_names = {
            "name",
            "room_name",
            "name_without_policy",
            "room_surface_in_feet2",
            "available",
            "room_count",
            "product_price_breakdown",
        }
        for match in re.finditer(r"(?P<quote>['\"])(?P<literal>[^'\"]+)(?P=quote)\s+in\s+(?P<rhs>[^\n:]+)", code or ""):
            literal = match.group("literal")
            rhs = match.group("rhs")
            if literal in key_names:
                continue
            if (
                "['name']" in rhs
                or '["name"]' in rhs
                or ".get('name'" in rhs
                or '.get("name"' in rhs
            ):
                return True
        for match in re.finditer(r"(?P<lhs>\b[a-zA-Z_]\w*(?:\.\w+\(\))?)\s+in\s+(?P<rhs>[^\n:]+)", code or ""):
            rhs = match.group("rhs")
            if (
                "['name']" in rhs
                or '["name"]' in rhs
                or ".get('name'" in rhs
                or '.get("name"' in rhs
            ):
                return True
        return False

    def _validate_generated_code(self, code, question):
        q = self._norm_text(question)
        code_text = code or ""

        if (
            ("available rooms of the kind" in q or "area in square feet" in q)
            and re.search(r"['\"](?:room_name|name_without_policy)['\"]", code_text)
        ):
            raise ValueError(
                "Semantic validation failed: exact room-name tasks must match room['name']; "
                "do not read room_name or name_without_policy."
            )

        if (
            "available rooms of the kind" in q
            and "available" in code_text
            and "room_count" not in code_text
            and re.search(r"\.get\(\s*['\"]available['\"]\s*,", code_text)
        ):
            raise ValueError(
                "Semantic validation failed: use room.get('available', room.get('room_count')), "
                "not a default zero/string when `available` is absent."
            )

        if "travel insurance plan" in q and re.search(r"for\s+\w+\s+in\s+[^\n:]*options", code_text):
            raise ValueError(
                "Semantic validation failed: travelInsurance['options'] is a dict, not a list. "
                "Do not iterate over it; access options['type'] and options['priceBreakdown']['total'] directly."
            )

    def _validate_generated_solution(self, code, question, api_response, output):
        """Raise on common semantic mistakes so the debug loop can repair code."""
        q = self._norm_text(question)
        code_text = code or ""

        if "all inclusive cost" in q and "all_inclusive_amount" not in code_text:
            raise ValueError(
                "Semantic validation failed: the question asks for all inclusive cost. "
                "Use product_price_breakdown['all_inclusive_amount']['value'], not gross_amount_per_night or display strings."
            )

        if "gross rate less than" in q and "gross_amount_per_night" not in code_text:
            raise ValueError(
                "Semantic validation failed: the question asks for gross rate less than a threshold. "
                "Use product_price_breakdown['gross_amount_per_night']['value']."
            )
        if (
            ("output a comma separated list of room names" in q or q.startswith("get rooms with"))
            and re.search(r"['\"](?:room_name|name_without_policy)['\"]", code_text)
        ):
            raise ValueError(
                "Semantic validation failed: list answers must append exact room['name'] values. "
                "Do not use room_name or name_without_policy because they omit policy text needed by the answer."
            )

        if "travel insurance plan" in q:
            if re.search(r"for\s+\w+\s+in\s+.*options", code_text):
                raise ValueError(
                    "Semantic validation failed: travelInsurance['options'] is a dict, not a list. "
                    "Do not iterate over it; access options['type'] and options['priceBreakdown']['total'] directly."
                )
            if "nanos" not in code_text:
                raise ValueError(
                    "Semantic validation failed: insurance totals require units plus nanos / 1_000_000_000."
                )
            options = api_response.get("data", {}).get("travelInsurance", {}).get("options", {}) if isinstance(api_response, dict) else {}
            total = options.get("priceBreakdown", {}).get("total", {}) if isinstance(options, dict) else {}
            units = total.get("units")
            nanos = total.get("nanos", 0)
            if nanos and units is not None:
                output_text = str(output)
                if re.search(rf"\b{re.escape(str(units))}\b\s*$", output_text) and "." not in output_text.split(str(units), 1)[-1]:
                    raise ValueError(
                        "Semantic validation failed: insurance output only contains whole units even though nanos is nonzero. "
                        "Return currencyCode plus units + nanos / 1_000_000_000."
                    )

        if "luggage allowance" in q and (
            "luggageType" not in code_text or "maxWeightPerPiece" not in code_text or "massUnit" not in code_text
        ):
            raise ValueError(
                "Semantic validation failed: luggage answer must include luggageType, maxWeightPerPiece, and massUnit."
            )

        if "available rooms of the kind" in q:
            room_name = self._extract_room_name(question, "What is the total number of available rooms of the kind ")
            room = self._find_room_by_question_name(api_response, room_name)
            if room is not None:
                if room_name and self._norm_text(room_name) not in self._norm_text(code_text):
                    raise ValueError(
                        "Semantic validation failed: use the full parsed room name from the question as the match target. "
                        "Do not match only a shorter substring such as 'King Room' or 'Double Room'."
                    )
                if output in (None, "", "None", "none"):
                    raise ValueError(
                        "Semantic validation failed: an exact room `name` exists in data['available']; "
                        "match against room['name'] and return room['available'] if present, else room['room_count']."
                    )
                if str(output).strip() == "0":
                    raise ValueError(
                        "Semantic validation failed: the parsed room exists in data['available'], so a zero output usually means the code matched the wrong field or used substring logic. "
                        "Use exact equality against room['name'] and return its availability field."
                    )
                if self._uses_substring_name_match(code_text):
                    raise ValueError(
                        "Semantic validation failed: exact room-name tasks must use normalized equality, not substring `in` matching."
                    )
                if re.search(r"\b(len|sum)\s*\(|\+=\s*1", code_text):
                    raise ValueError(
                        "Semantic validation failed: this task asks for the room's availability field, not the number of matching records. "
                        "Find the exact room by `name`, then return room['available'] or room['room_count']."
                    )
                if "available" not in code_text and "room_count" not in code_text:
                    raise ValueError(
                        "Semantic validation failed: this is not a count of matching records. "
                        "After finding the exact room by `name`, return its `available` or `room_count` field."
                    )

        if "area in square feet" in q:
            room_name = self._extract_room_name(
                question,
                "What is the area in square feet of ",
                "? Include just the number and not the unit.",
            )
            room = self._find_room_by_question_name(api_response, room_name)
            if room is not None:
                if room_name and self._norm_text(room_name) not in self._norm_text(code_text):
                    raise ValueError(
                        "Semantic validation failed: use the full parsed room name from the question as the match target. "
                        "Do not match only a shorter substring."
                    )
                if output in (None, "", "None", "none"):
                    raise ValueError(
                        "Semantic validation failed: an exact room `name` exists in data['available']; "
                        "match room['name'] and return room['room_surface_in_feet2']."
                    )
                if self._uses_substring_name_match(code_text):
                    raise ValueError(
                        "Semantic validation failed: exact room-name tasks must use normalized equality, not substring `in` matching."
                    )
                if re.search(r"\b(int|round)\s*\(", code_text):
                    raise ValueError(
                        "Semantic validation failed: area questions require the exact `room_surface_in_feet2` value. "
                        "Do not round or cast it to int."
                    )
                if "room_surface_in_feet2" not in code_text:
                    raise ValueError(
                        "Semantic validation failed: area questions must return room['room_surface_in_feet2'], "
                        "not price, size labels, or a converted value."
                    )

    def format_prompt(self, sample):
        question = sample.get("question", "")
        raw_json_str = _load_json_for_tooljson(sample, self.base_repo_dir)
        api_query = sample.get("api_query", "")
        task_name = sample.get("task", "")
        task_type = sample.get("task_type", "")

        target_payload = self._extract_target_json(raw_json_str, api_query)
        sample_data = self._payload_excerpt(target_payload)
        shape_summary = self._shape_summary(target_payload)
        list_paths = self._primary_list_paths(target_payload)
        candidates = self._candidate_value_summary(target_payload)
        list_paths_text = "\n".join(f"- {path}" for path in list_paths) if list_paths else "- No list paths found."
        question_guidance = self._question_guidance(sample)

        prompt = (
            "You are writing a small, deterministic Python extractor for one JSON QA task.\n"
            "Return ONLY Python code that defines `solve(data)`. No markdown, no explanation.\n\n"
            "Rules for `solve(data)`:\n"
            "- Use the provided JSON structure; do not invent fields.\n"
            "- Search all records, not only data[0].\n"
            "- Do not iterate directly over a dict root. Use the listed list path, for example data['available'] or data['data']['result'].\n"
            "- Match names, IDs, dates, and filters from the question case-insensitively unless exact IDs are requested.\n"
            "- Normalize both sides before text comparison: `needle.lower() in value.lower()` or equality on `.lower()`.\n"
            "- Do not add extra assumptions from field names that are not in the question, such as roomtype_id, free_cancellation, secret_deal, or dormitory.\n"
            "- For booking room names, prefer the `name` field when present; `room_name` can differ from the question wording.\n"
            "- For 'total number of available rooms of the kind X', do not count records. Find the room whose `name` equals X, then return `available` or `room_count`.\n"
            "- For 'area in square feet of X', find the room whose `name` equals X, then return `room_surface_in_feet2` exactly; do not round or cast to int.\n"
            "- For room-list answers, append `room['name']` exactly, not `room_name` or `name_without_policy`.\n"
            "- For all-inclusive cheapest-room questions, compute min over `product_price_breakdown['all_inclusive_amount']['value']`.\n"
            "- For gross-rate threshold questions, compare `product_price_breakdown['gross_amount_per_night']['value']` to the threshold.\n"
            "- For money/rates/prices, prefer numeric `value` fields over display strings like `amount_rounded` or `amount_unrounded`.\n"
            "- Return only the answer value from solve(data); no explanatory sentence.\n"
            "- For counts, return an int or numeric string.\n"
            "- For list questions, return one comma-separated string using exactly ', ' between items.\n"
            "- For missing data, return None.\n"
            "- The function must be self-contained and safe on missing keys.\n\n"
            f"Task name: {task_name}\n"
            f"Task type: {task_type}\n"
            f"Parsed task guidance:\n{question_guidance}\n\n"
            f"Important list paths:\n{list_paths_text}\n\n"
            f"Candidate text/id values:\n{candidates}\n\n"
            f"JSON shape summary:\n{shape_summary}\n\n"
        )
        if api_query:
            prompt += f"API Query Context: {api_query}\n"

        prompt += (
            f"JSON payload excerpt:\n{sample_data}\n\n"
            f"Question: {question}\n\n"
            "Python code:"
        )
        return [{"role": "user", "content": prompt}]

    def execute_debug_loop(self, client, sample_id, sample):
        from extensions.self_debug.debug_loop import debug_loop
        import datetime
        import math
        import statistics

        question = sample.get("question", "")
        api_query = sample.get("api_query", "")
        raw_json_str = _load_json_for_tooljson(sample, self.base_repo_dir)

        # Use the actual payload!
        api_response = self._extract_target_json(raw_json_str, api_query)

        prompt_orig = self.format_prompt(sample)[0]['content']
        schema = self._shape_summary(api_response, max_paths=120)

        def model_generate(prompt, temperature=0.3, **kwargs):
            # No stop sequences: code blocks are often truncated by generic stop tokens.
            res = client.generate(
                prompt,
                stop=[],
                temperature=temperature,
                max_tokens=1800,
                max_retries=2,
            )
            return self._extract_python_code(res)

        def execute_code(code, data):
            self._validate_generated_code(code, question)
            env = {
                "json": json,
                "re": re,
                "math": math,
                "statistics": statistics,
                "datetime": datetime,
            }
            try:
                # Use one namespace. With separate globals/locals, imports land in
                # locals but functions resolve globals, so generated code loses imports.
                exec(code, env, env)
            except Exception as e:
                # Compile errors etc
                raise e
            if 'solve' not in env:
                raise ValueError("Function 'solve(data)' not found in generated code")
            output = env['solve'](data)
            self._validate_generated_solution(code, question, data, output)

            return output

        # Run loop
        final_output, history = debug_loop(
            model_generate=model_generate,
            execute_code=execute_code,
            prompt_orig=prompt_orig,
            api_response=api_response,
            question=question,
            schema=schema,
            max_rounds=6,
            error_levels=(1, 2, 3), # Catch all
            temperature=0.3
        )

        if final_output is None or final_output == "":
            # fallback to history if exists to see what happened
            if history and "output" in history[-1] and history[-1]["output"] is not None:
                final_output = history[-1]["output"]
            else:
                if history:
                    return f"ERROR: {history[-1].get('error', {}).get('message', 'Unknown Error')} \nCODE: {history[-1].get('code', '')}"
                return ""

        q_norm = self._norm_text(question)
        if isinstance(final_output, (list, dict)):
            return json.dumps(final_output)
        float_style_numeric = any(
            marker in q_norm
            for marker in (
                "average price",
                "price",
                "cost",
                "rate",
                "vat",
            )
        ) or q_norm.startswith("what is the rating of") or "cleanliness rating" in q_norm
        force_float_style = (
            "average price" in q_norm
            or q_norm.startswith("what is the rating of")
            or "cleanliness rating" in q_norm
        )
        if isinstance(final_output, float) and float_style_numeric:
            return str(final_output)
        if isinstance(final_output, int) and force_float_style:
            return str(float(final_output))
        # Force integer/float formatting without decimals if integer
        if isinstance(final_output, (int, float)):
            try:
                if final_output == int(final_output):
                    return str(int(final_output))
            except Exception:
                pass
        if isinstance(final_output, str):
            final_output = final_output.strip()
            if "comma separated list" in q_norm or "comma-separated list" in q_norm:
                final_output = re.sub(r"\s*,\s*", ", ", final_output)
            if re.fullmatch(r"[-+]?\d+\.\d+", final_output) and float_style_numeric:
                final_output = str(float(final_output))
            if "luggage allowance" in q_norm:
                final_output = re.sub(r"(\d(?:\.\d+)?)\s+([A-Z]{2,})\s*$", r"\1\2", final_output)
            keep_units = any(
                marker in q_norm
                for marker in (
                    "travel insurance plan",
                    "luggage allowance",
                    "currency",
                    "price",
                    "cost",
                    "rate",
                )
            )
            if q_norm.startswith("what is") and "output a comma separated list" not in q_norm and not keep_units:
                numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", final_output.replace(",", ""))
                if len(numbers) == 1 and final_output.strip() != numbers[0]:
                    return numbers[0]
        return str(final_output).strip()


# =============================================================================
# Registry: map strategy name -> loader classes

# =============================================================================
STRATEGIES = {
    "s1_prompt": {
        "toolbench": S1_ToolBench,
        "complexfunc": S1_ComplexFunc,
        "tooljson": S1_ToolJSON,
        "label": "Strategy 1: Prompt Rewrite",
    },
    "s2_compress": {
        "toolbench": S2_ToolBench,
        "complexfunc": S2_ComplexFunc,
        "tooljson": S2_ToolJSON,
        "label": "Strategy 2: Tool Compression",
    },
    "s3_cot": {
        "toolbench": S3_ToolBench,
        "complexfunc": S3_ComplexFunc,
        "tooljson": S3_ToolJSON,
        "label": "Strategy 3: Chain-of-Thought",
    },
    "s4_twostage": {
        "toolbench": S4_ToolBench,
        "complexfunc": S4_ComplexFunc,
        "tooljson": S4_ToolJSON,
        "label": "Strategy 4: Two-Stage LLM",
    },
    "s5_fewshot": {
        "toolbench": S5_ToolBench,
        "complexfunc": S5_ComplexFunc,
        "tooljson": S5_ToolJSON,
        "label": "Strategy 5: Few-Shot Examples",
    },
    "s6_context": {
        "toolbench": S6_ToolBench,
        "complexfunc": S6_ComplexFunc,
        "tooljson": S6_ToolJSON,
        "label": "Strategy 6: Context-Aware Pipeline",
    },
    "s7_selfdebug": {
        "toolbench": S6_ToolBench, # Placeholder
        "complexfunc": S6_ComplexFunc, # Placeholder
        "tooljson": S7_SelfDebug_ToolJSON,
        "label": "Strategy 7: Self-Debug Code Execution",
    },
}
