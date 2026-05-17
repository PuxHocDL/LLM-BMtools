"""
Deploy Llama-4-maverick-17b on Modal using vLLM.

Usage:
    modal serve deploy_llama4.py
"""

import modal

IMAGE = modal.Image.from_registry("vllm/vllm-openai:latest")
MODEL_NAME = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
GPU = "a100-80gb"  # Large model needs more memory
APP_NAME = "llama-4-maverick-17b-instruct"


@modal.stub(
    name=APP_NAME,
    image=IMAGE,
    gpu=GPU,
    timeout=600,
)
def serve_model():
    """Run vLLM OpenAI-compatible API server."""
    import subprocess

    print(f"Starting vLLM server for {MODEL_NAME}...")
    print(f"GPU: {GPU}")

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "128000",
        "--tensor-parallel-size", "2",  # For large models
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.9",
    ]

    subprocess.run(cmd)


if __name__ == "__main__":
    print("="*60)
    print("Modal vLLM Deployment: Llama-4-maverick-17b")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"App Name: {APP_NAME}")
    print(f"GPU: {GPU}")
    print()
    print("Deploying...")
