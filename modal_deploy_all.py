"""
Complete Modal deployment and evaluation script.

Usage:
    1. Deploy models first: python modal_deploy_all.py
    2. Get deployment URLs from Modal output
    3. Update API_BASE in config_modal.yaml with the deployed URLs
    4. Run evaluation: python evaluate_on_modal.py --model granite-33-8b
"""

import modal
import subprocess
import time

# vLLM image
VLLM_IMAGE = "vllm/vllm-openai:latest"

# Model configurations
MODELS = {
    "granite-33-8b": {
        "model_id": "ibm-granite/granite-3.3-8b-instruct",
        "app_name": "granite-33-8b-server",
        "gpu": "a10g",
    },
    "llama-4-maverick-17b": {
        "model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "app_name": "llama-4-maverick-17b-server",
        "gpu": "a100-80gb",
    },
    "gpt-oss-20b": {
        "model_id": "openai/gpt-oss-20b",
        "app_name": "gpt-oss-20b-server",
        "gpu": "a100-40gb",
    }
}


def create_model_app(model_key: str, config: dict):
    """Create a Modal app for serving the model."""

    stub = modal.Stub(config["app_name"])

    @stub.function(
        image=VLLM_IMAGE,
        gpu=config["gpu"],
        timeout=600,
        memory=1024 * 1024 * 1024,  # 1GB
    )
    def serve():
        """Start vLLM server."""
        import subprocess

        print(f"Starting vLLM server for {config['model_id']}")

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", config["model_id"],
            "--host", "0.0.0.0",
            "--port", "8000",
            "--max-model-len", "128000",
        ]

        subprocess.run(cmd)

    @stub.local_entrypoint()
    def deploy():
        """Deploy the model server."""
        print(f"Deploying {config['app_name']}...")
        return stub.serve()

    return stub


def deploy_all_models():
    """Deploy all models and return their URLs."""

    deployed_urls = {}

    for model_key, config in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Deploying: {model_key}")
        print(f"{'='*60}")
        print(f"Model ID: {config['model_id']}")
        print(f"App Name: {config['app_name']}")
        print(f"GPU: {config['gpu']}")
        print()

        try:
            stub = create_model_app(model_key, config)

            # Deploy using Modal CLI
            result = subprocess.run(
                ["python", "-m", "modal", "deploy", "vllm_serve.py",
                 "--name", config["app_name"]],
                capture_output=True,
                text=True,
                timeout=120
            )

            print("Deployment output:")
            print(result.stdout)

            if result.returncode == 0:
                # Extract URL from output (Modal prints the URL)
                # The URL format is usually: https://<app-name>--<workspace>.modal.run
                deployed_urls[model_key] = f"https://{config['app_name']}.modal.run"
                print(f"✓ Deployed: {model_key}")
            else:
                print(f"✗ Failed: {model_key}")
                print("Error:", result.stderr)

        except Exception as e:
            print(f"✗ Error deploying {model_key}: {e}")

    print(f"\n{'='*60}")
    print("Deployment Summary")
    print(f"{'='*60}")

    for model_key, url in deployed_urls.items():
        print(f"{model_key}: {url}")

    return deployed_urls


if __name__ == "__main__":
    print("Modal LLM Deployment & Evaluation")
    print("="*50)
    print("\nThis script will:")
    print("1. Deploy 3 models on Modal (granite-33-8b, llama-4-maverick-17b, gpt-oss-20b)")
    print("2. Print the deployment URLs")
    print("3. Generate config_modal.yaml with the correct URLs")
    print("\nAfter deployment, you can run evaluation with:")
    print("  python main.py --dataset tooljson --model granite-33-8b --limit 100")
    print()

    deploy_all_models()
