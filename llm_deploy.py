"""
Modal app to serve LLM models with OpenAI-compatible API.
Supports Granite-3.3-8b, Llama-4-maverick-17b, and GPT-oss-20b.
"""

import modal
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
import time

# Image with GPU for model inference
IMAGE = "nvidia/cuda:12.1.0-runtime"

# App configuration
app = modal.App("llm-tooljson-evaluation")


# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 1024
    stop: Optional[List[str]] = None
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict


class APIResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]


# Model containers with appropriate GPU settings
def get_container_for_model(model_name: str):
    """Return GPU container configuration for the model."""
    # Adjust GPU type and memory based on model size
    return {
        "image": IMAGE,
        "gpu": modal.gpu.L4(count=1),  # L4 GPU for smaller models, adjust if needed
        "timeout": 600,
    }


# Model loading function (cached per model)
@modal.cls(gpu="any", image=IMAGE, concurrency_limit=3)
class ModelContainer:
    """Container for serving LLM models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    @modal.enter()
    def load_model(self):
        """Load the model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded: {self.model_name}")

    @modal.method()
    def generate(self, messages: list, temperature: float = 0.0,
                 max_tokens: int = 1024, stop: list = None) -> str:
        """Generate text from the model."""
        import torch

        # Format messages
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        return response


# FastAPI app wrapper for OpenAI-compatible API
@modal.web_endpoint(method="POST")
async def chat_completions(request: ChatCompletionRequest) -> APIResponse:
    """OpenAI-compatible chat completions endpoint."""

    # Create model instance name from model_id
    model_name = request.model

    # Get the model container
    try:
        # Create or get model container for this model
        with modal.enable_output():
            # We'll use a simple approach - create containers per model on demand
            model_container = ModelContainer(model_name)

            # Generate response
            response_text = model_container.generate.remote(
                messages=[[m.role, m.content] for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop or []
            )

            # Format response
            import time
            return APIResponse(
                id=f"chatcmpl-{int(time.time() * 1000)}",
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }]
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


# Local entrypoint for testing
@modal.local_entrypoint()
def test():
    """Test the deployment."""
    print("Testing model deployment...")

    # Test with a simple query
    result = chat_completions.remote(
        ChatCompletionRequest(
            model="ibm-granite/granite-3.3-8b-instruct",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=50
        )
    )

    print("Response:", result.choices[0]["message"]["content"])
    return result


if __name__ == "__main__":
    # For local testing with FastAPI
    import uvicorn

    @app.function()
    @modal.web_server(8000)
    def fastapi_app():
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        web_app = FastAPI(title="LLM Evaluation API")

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.post("/v1/chat/completions")
        async def api_chat(request: ChatCompletionRequest):
            return await chat_completions(request)

        return web_app

    # Run the server
    # Note: This would be started with: modal serve llm_deploy.py
    pass
