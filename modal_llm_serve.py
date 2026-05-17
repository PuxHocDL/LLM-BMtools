"""
Simple Modal app to serve a single LLM model with OpenAI-compatible API.
Deploy one instance per model.
"""

import modal
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time

# Use Modal's vLLM image for model serving
IMAGE = "vllm/vllm-openai:latest"

app = modal.App("llm-evaluation")

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 1024
    stream: bool = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]

# Keep the app alive
@modal.function(image=IMAGE, gpu="any", timeout=600)
def keep_alive():
    """Keep the container alive."""
    import time
    time.sleep(600)
    return "Alive"

# FastAPI web endpoint
@modal.web_endpoint(method="POST", image=IMAGE, gpu="any", timeout=600)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint."""
    from openai import OpenAI

    # Use Modal's internal OpenAI client
    # The model will be served by vLLM
    client = OpenAI(
        base_url="http://localhost:8000/v1",  # vLLM OpenAI-compatible server
        api_key="dummy"
    )

    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[[m.role, m.content] for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return ChatCompletionResponse(
            id=response.id,
            object=response.object,
            created=response.created,
            model=response.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": response.choices[0].message.role,
                    "content": response.choices[0].message.content
                },
                "finish_reason": response.choices[0].finish_reason
            }]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Local entrypoint for deployment
@app.local_entrypoint()
def deploy():
    """Deploy the app."""
    print("Deploying LLM evaluation API...")
    return "Deployed successfully"

if __name__ == "__main__":
    deploy()
