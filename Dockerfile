# Pointwise decision server (POST /v1/systemone) on one NVIDIA GPU.
# Build:  docker build -t pointwise .
# Run:    see README "Run it in Docker". Weights, metrics and the HF model cache are mounted at run time, never baked in.
#
# The CUDA base only supplies the user-space CUDA libraries' environment; the torch wheel brings its own CUDA 12.8
# runtime, so the base's 12.8 must match the wheel's cu128 and the host driver must support 12.8 (driver >= 570).
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Dependencies before code: these layers are cached and only rebuilt when the pins change.
RUN pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu128
COPY requirements-serve.txt /app/requirements-serve.txt
RUN pip install -r /app/requirements-serve.txt

COPY src /app/src
COPY configs /app/configs
WORKDIR /app

# No fixed uid: the Ubuntu 24.04 base already has user "ubuntu" at 1000. The cache dir is created here so a named volume
# mounted on it starts owned by app, not root (Docker copies the image directory's owner into a new volume).
RUN useradd --create-home app && mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache
USER app
ENV HF_HOME=/home/app/.cache/huggingface POINTWISE_HOST=0.0.0.0 POINTWISE_PORT=8000
EXPOSE 8000

# /healthz is liveness only (no auth, no GPU); readiness for a load balancer is /readyz. The start period covers the
# first start's ~3.4 GB base-model download, before which nothing listens.
HEALTHCHECK --interval=15s --timeout=3s --start-period=900s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('POINTWISE_PORT', '8000')}/healthz\", timeout=2)"]

# Exec form: python is PID 1 and receives SIGTERM itself, so serve_main's drain runs (a shell wrapper would swallow it).
CMD ["python", "-m", "src.serve_main"]
