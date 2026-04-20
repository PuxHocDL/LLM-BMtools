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
        raw_json = _load_json_for_tooljson(sample, self.base_repo_dir)
        json_content = JSONPruner.prune(raw_json, question, top_k=50)

        prompt = (
            "You are a data extraction assistant. Given API response data and a question, "
            "output ONLY the answer value. No explanation, no sentences.\n"
            "- If the answer is a number, output just the number.\n"
            "- If the answer is a name/type/date, output just that value.\n"
            "- If multiple values, output comma-separated.\n\n"
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
        raw_json = _load_json_for_tooljson(sample, self.base_repo_dir)
        json_content = JSONPruner.prune(raw_json, question, top_k=50)

        prompt = (
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
        raw_json = _load_json_for_tooljson(sample, self.base_repo_dir)
        json_content = JSONPruner.prune(raw_json, question, top_k=50)

        prompt = (
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
        raw_json = _load_json_for_tooljson(sample, self.base_repo_dir)
        pruned = JSONPruner.prune(raw_json, question, top_k=50)

        # Stage 1: Ask LLM to extract just the relevant data
        stage1_prompt = (
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
        raw_json = _load_json_for_tooljson(sample, self.base_repo_dir)
        json_content = JSONPruner.prune(raw_json, question, top_k=50)

        prompt = (
            "You are a data extraction assistant.\n\n"
            "--- EXAMPLE ---\n"
            f"Data: {_TOOLJSON_EXAMPLE['data']}\n"
            f"Question: {_TOOLJSON_EXAMPLE['question']}\n"
            f"Answer: {_TOOLJSON_EXAMPLE['answer']}\n"
            "--- END EXAMPLE ---\n\n"
            f"Data:\n{json_content}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return [{"role": "user", "content": prompt}]


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
}
