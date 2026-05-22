import yaml
from openai import OpenAI
import httpx
import os
import time
import concurrent.futures

class LLMClient:
    def __init__(self, config_path="config.yaml", agent_name="qwen"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        if agent_name not in self.config["agents"]:
            raise ValueError(f"Agent '{agent_name}' not found in config.")
            
        agent_cfg = self.config["agents"][agent_name]
        self.agent_name = agent_name
        self.model_name = agent_cfg["model"]
        self.api_base = agent_cfg["api_base"]
        self.api_key = agent_cfg["api_key"]
        self.temperature = agent_cfg.get("temperature", 0.0)
        self.max_steps = agent_cfg.get("max_steps", 30)
        self.timeout = agent_cfg.get("timeout", 180.0)

        # Initialize OpenAI client with both per-operation and total timeout
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=httpx.Timeout(self.timeout, connect=5.0),
            max_retries=0
        )

    def _call_api(self, use_completions=False, **kwargs):
        """Call API with a hard total timeout to prevent hanging."""
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        api_func = self.client.completions.create if use_completions else self.client.chat.completions.create
        future = pool.submit(api_func, **kwargs)
        try:
            return future.result(timeout=self.timeout)
        except (TimeoutError, concurrent.futures.TimeoutError):
            future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(f"API call exceeded hard timeout of {self.timeout}s")
        finally:
            pool.shutdown(wait=False)

    def generate(
        self,
        prompt_or_messages,
        system_message="You are a helpful assistant.",
        max_retries=1,
        stop=None,
        enable_thinking=False,
        max_tokens=None,
        temperature=None,
        top_p=None,
    ):
        is_base_model = ("llama" in self.model_name.lower() and "instruct" not in self.model_name.lower())

        if isinstance(prompt_or_messages, str):
            if is_base_model:
                raw_prompt = f"{system_message}\n\n{prompt_or_messages}\n\nAssistant:"
            else:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt_or_messages}
                ]
        else:
            if is_base_model:
                raw_prompt = ""
                for msg in prompt_or_messages:
                    raw_prompt += f"{msg['content']}\n\n"
                raw_prompt += "Assistant:"
            else:
                messages = prompt_or_messages

        if stop is None:
            stop = ["\nObservation:", "\nCall result:", "Observation:", "Call result:", "\nThought:", "\n{\""]
            
        supports_thinking = (
            "qwen" in self.agent_name.lower()
            or "qwen" in self.model_name.lower()
        )

        kwargs = dict(
            model=self.model_name,
            temperature=(
                max(self.temperature, 0.6)
                if enable_thinking and supports_thinking and temperature is None
                else self.temperature if temperature is None
                else temperature
            ),
            max_tokens=max_tokens or 2048,
        )
        if top_p is not None:
            kwargs["top_p"] = top_p
        
        if is_base_model:
            kwargs["prompt"] = raw_prompt
        else:
            kwargs["messages"] = messages
            
        if enable_thinking and supports_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
            
        if stop:
            kwargs["stop"] = stop
            
        for attempt in range(max_retries):
            try:
                response = self._call_api(use_completions=is_base_model, **kwargs)
                if is_base_model:
                    content = response.choices[0].text
                else:
                    content = response.choices[0].message.content
                return content if content is not None else ""
            except Exception as e:
                err_str = str(e)
                if "400" in err_str and "context length" in err_str.lower():
                    return "__ERROR_CONTEXT_LENGTH__"
                
                if attempt == max_retries - 1:
                    return f"__ERROR_API__: {err_str}"
                time.sleep(2 ** attempt) # Exponential backoff: 1s, 2s, 4s...
            
    def generate_json(self, prompt, system_message="You are a helpful assistant. Respond in JSON format.", max_retries=2):
        is_base_model = ("llama" in self.model_name.lower() and "instruct" not in self.model_name.lower())

        for attempt in range(max_retries):
            try:
                if is_base_model:
                    raw_prompt = f"{system_message}\n\n{prompt}\n\n```json\n"
                    response = self._call_api(
                        use_completions=True,
                        model=self.model_name,
                        prompt=raw_prompt,
                        temperature=self.temperature,
                        max_tokens=2048,
                        stop=["```"]
                    )
                    content = response.choices[0].text
                    if content and not content.lstrip().startswith("{"):
                        content = "{" + content
                    return content if content is not None else "{}"
                else:
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ]
                    kwargs = dict(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=2048
                    )
                    
                    if "llama" not in self.model_name.lower():
                        kwargs["response_format"] = {"type": "json_object"}
                        
                    response = self._call_api(use_completions=False, **kwargs)
                    content = response.choices[0].message.content
                    return content if content is not None else "{}"
                    
            except Exception as e:
                err_str = str(e)
                if "400" in err_str and "context length" in err_str.lower():
                    return "{}"
                    
                print(f"Warning: API attempt {attempt + 1} failed for {self.model_name}: {err_str}")
                if attempt == max_retries - 1:
                    return "{}"
                time.sleep(2 ** attempt)
