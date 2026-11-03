"""Pointwise on Modal: one T4, scales to zero. Run with `modal serve deploy/modal_app.py` (temporary URL) or
`modal deploy deploy/modal_app.py` (permanent). Needs the Secret `pointwise-key` (POINTWISE_API_KEY) and the Volume
`pointwise-weights` (state.pt and metrics.json from the same run). See docs/modal.md."""
import subprocess

import modal

MODEL = "Qwen/Qwen3-1.7B-Base"  # same as src/train.py MODEL_NAME

app = modal.App("pointwise")
weights = modal.Volume.from_name("pointwise-weights")


def fetch_base_model():
    # Runs once at image build time, so the 3.4 GB download is part of the image, not of every cold start.
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)


image = (
    modal.Image.debian_slim(python_version="3.12")
    # Same torch as the Dockerfile: the CUDA 12.8 wheel.
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install_from_requirements("requirements-serve.txt")
    .env(
        {
            "HF_HOME": "/hf",
            "POINTWISE_HOST": "0.0.0.0",  # Modal can only reach a server bound to 0.0.0.0
            "POINTWISE_PORT": "8000",
            "POINTWISE_WEIGHTS": "/weights/state.pt",
            "POINTWISE_METRICS": "/weights/metrics.json",
        }
    )
    .run_function(fetch_base_model)
    .add_local_dir("src", "/app/src")  # local-file lines stay last
    .add_local_dir("configs", "/app/configs")
)


@app.function(
    image=image,
    gpu="T4",  # the only GPU measured (reports/pointwise-load.md)
    volumes={"/weights": weights},
    secrets=[modal.Secret.from_name("pointwise-key")],  # provides POINTWISE_API_KEY
    min_containers=0,  # scale to zero: no traffic, no cost
    max_containers=1,  # the server has one GPU lock and one cache; also a cost cap
    scaledown_window=60,  # idle seconds before the container stops
    timeout=3600,
)
@modal.concurrent(max_inputs=16)  # matches POINTWISE_MAX_QUEUE=16; beyond that the server answers 503 itself
@modal.web_server(8000, startup_timeout=300)
def serve():
    subprocess.Popen(["python", "-m", "src.serve_main"], cwd="/app")
