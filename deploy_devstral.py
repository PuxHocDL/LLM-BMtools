"""
Deploy Devstral Small 2507 on Modal as an OpenAI-compatible vLLM API.

Usage:
    modal deploy deploy_devstral.py
"""

import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

MODEL_NAME = "mistralai/Devstral-Small-2507"
GPU = "H200:1"
APP_NAME = "devstral-small-2507"
VLLM_PORT = 8000
MAX_MODEL_LEN = "128000"
FAST_BOOT = True

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=vllm_image,
    gpu=GPU,
    timeout=10 * 60,
    scaledown_window=15 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=300)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        MAX_MODEL_LEN,
        "--gpu-memory-utilization",
        "0.95",
    ]

    if FAST_BOOT:
        cmd.append("--enforce-eager")
    else:
        cmd.append("--no-enforce-eager")

    print("Starting vLLM server with command:")
    print(" ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


if __name__ == "__main__":
    print("=" * 60)
    print("Modal vLLM Deployment - Devstral Small 2507")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"App Name: {APP_NAME}")
    print(f"GPU: {GPU}")
    print(f"Max model len: {MAX_MODEL_LEN}")
    print("Max concurrent requests: 300")
    print(f"Fast boot: {FAST_BOOT}")
    print()
    print("Deploying with: modal deploy deploy_devstral.py")
    print("=" * 60)
