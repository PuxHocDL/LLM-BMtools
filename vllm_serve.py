"""
Modal vLLM server for serving HuggingFace models with OpenAI-compatible API.
Supports: Granite-3.3-8b, Llama-4-maverick-17b, GPT-oss-20b

Usage:
    modal serve vllm_serve.py --model-name ibm-granite/granite-3.3-8b-instruct
"""

import argparse
import modal
from typing import Optional

# vLLM image with GPU support
VLLM_IMAGE = "vllm/vllm-openai:latest"

# Model configurations
MODEL_CONFIGS = {
    "granite-33-8b": {
        "model": "ibm-granite/granite-3.3-8b-instruct",
        "gpu": "a10g",  # Adjust GPU type based on availability
        "max_model_len": 128000,
    },
    "llama-4-maverick-17b": {
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "gpu": "a100-80gb",  # Large model needs more memory
        "max_model_len": 128000,
    },
    "gpt-oss-20b": {
        "model": "openai/gpt-oss-20b",
        "gpu": "a100-40gb",
        "max_model_len": 128000,
    }
}


def create_vllm_serve(model_name: str, gpu: str, max_model_len: int):
    """Create a vLLM serve app for the model."""

    stub = modal.Stub(name=f"vllm-{model_name.replace('/', '-').replace('_', '-')}")

    @stub.function(
        image=VLLM_IMAGE,
        gpu=gpu,
        timeout=600,
        concurrency_limit=1,
    )
    @modal.web_server(8000)
    def serve():
        """Start the vLLM server."""
        import subprocess
        import sys

        print(f"Starting vLLM server for {model_name}")
        print(f"GPU: {gpu}")
        print(f"Max model length: {max_model_len}")
        print(f"OpenAI-compatible API available at: https://<deployed-url>/v1")

        # Start vLLM OpenAI-compatible server
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--max-model-len", str(max_model_len),
        ]

        subprocess.run(cmd)

    return serve


def main():
    parser = argparse.ArgumentParser(description="Deploy vLLM server on Modal")
    parser.add_argument("--model",
                        choices=["granite-33-8b", "llama-4-maverick-17b", "gpt-oss-20b"],
                        required=True,
                        help="Model to deploy")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]

    print(f"Deploying {args.model} on Modal...")
    print(f"Model: {config['model']}")
    print(f"GPU: {config['gpu']}")
    print(f"Max model length: {config['max_model_len']}")
    print()

    # Create and deploy the app
    serve = create_vllm_serve(
        model_name=config['model'],
        gpu=config['gpu'],
        max_model_len=config['max_model_len']
    )

    # Deploy
    with modal.enable_output():
        print("Deploying...")
        # This will deploy and return the URL
        # In actual usage, you would run: modal serve vllm_serve.py --model granite-33-8b
        url = serve.deploy()
        print(f"Deployed at: {url}")
        print(f"API endpoint: {url}/v1")


if __name__ == "__main__":
    main()
