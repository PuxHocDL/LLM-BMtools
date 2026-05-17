"""
Deploy Granite-3.3-8b on Modal as OpenAI-compatible API.
Uses Modal's @web_server decorator to spawn vLLM's built-in server.

Usage:
    modal deploy deploy_granite.py
"""

import modal

# vLLM image with all dependencies - use Modal's fast uv_pip_install
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)

# Model configuration
MODEL_NAME = "ibm-granite/granite-3.3-8b-instruct"
MODEL_REVISION = "main"  # Granite doesn't pin specific revisions like Qwen
GPU = "H100:1"
APP_NAME = "granite-33-8b-instruct"
VLLM_PORT = 8000
# Full evaluation runs on H100 so vLLM can keep the long context window.
MAX_MODEL_LEN = "128000"

# Cache volumes for faster cold starts
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App(APP_NAME)

# Fast boot for development - set to False for production performance
FAST_BOOT = True

@app.function(
    image=vllm_image,
    gpu=GPU,
    timeout=10 * 60,  # 10 minutes for container start
    scaledown_window=15 * 60,  # 15 minutes before scaling down
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=500)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--served-model-name", "granite-33-8b",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--max-model-len", MAX_MODEL_LEN,
        "--gpu-memory-utilization", "0.95",
    ]

    # Fast boot mode - disable Torch compilation and CUDA graph capture
    if FAST_BOOT:
        cmd.append("--enforce-eager")
    else:
        cmd.append("--no-enforce-eager")

    print("Starting vLLM server with command:")
    print(" ".join(cmd))

    # Spawn vLLM server as subprocess
    subprocess.Popen(" ".join(cmd), shell=True)


if __name__ == "__main__":
    print("="*60)
    print("Modal vLLM Deployment - Granite-3.3-8b")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"App Name: {APP_NAME}")
    print(f"GPU: {GPU}")
    print(f"Max model len: {MAX_MODEL_LEN}")
    print("Max concurrent requests: 500")
    print(f"Fast boot: {FAST_BOOT}")
    print()
    print("Deploying with: modal deploy deploy_granite.py")
    print("="*60)
