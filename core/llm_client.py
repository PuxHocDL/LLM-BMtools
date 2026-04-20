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
        self.model_name = agent_cfg["model"]
        self.api_base = agent_cfg["api_base"]
        self.api_key = agent_cfg["api_key"]
        self.temperature = agent_cfg.get("temperature", 0.0)
        self.max_steps = agent_cfg.get("max_steps", 30)
        self.timeout = agent_cfg.get("timeout", 30.0)

        # Initialize OpenAI client with both per-operation and total timeout
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=httpx.Timeout(self.timeout, connect=5.0),
            max_retries=0
        )

    def _call_api(self, **kwargs):
        """Call API with a hard total timeout to prevent hanging."""
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self.client.chat.completions.create, **kwargs)
        try:
            return future.result(timeout=self.timeout)
        except (TimeoutError, concurrent.futures.TimeoutError):
            future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(f"API call exceeded hard timeout of {self.timeout}s")
        finally:
            pool.shutdown(wait=False)

    def generate(self, prompt_or_messages, system_message="You are a helpful assistant.", max_retries=1, stop=None, enable_thinking=False):
        if isinstance(prompt_or_messages, str):
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_or_messages}
            ]
        else:
            messages = prompt_or_messages

        if stop is None:
            stop = ["\nObservation:", "\nCall result:", "Observation:", "Call result:", "\nThought:", "\n{\""]
            
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=max(self.temperature, 0.6) if enable_thinking else self.temperature,
            max_tokens=512,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        if stop:
            kwargs["stop"] = stop
            
        for attempt in range(max_retries):
            try:
                response = self._call_api(**kwargs)
                content = response.choices[0].message.content
                return content if content is not None else ""
            except Exception as e:
                err_str = str(e)
                if "400" in err_str and "context length" in err_str.lower():
                    return "__ERROR_CONTEXT_LENGTH__"
                
                if attempt == max_retries - 1:
                    return f"__ERROR_API__: {err_str}"
                time.sleep(2 ** attempt) # Exponential backoff: 1s, 2s, 4s...
            
    def generate_json(self, prompt, system_message="You are a helpful assistant. Respond in JSON format.", max_retries=1):
        for attempt in range(max_retries):
            try:
                response = self._call_api(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    max_tokens=512
                )
                content = response.choices[0].message.content
                return content if content is not None else "{}"
            except Exception as e:
                err_str = str(e)
                if "400" in err_str and "context length" in err_str.lower():
                    return "{}"
                if attempt == max_retries - 1:
                    return "{}"
                time.sleep(2 ** attempt)
