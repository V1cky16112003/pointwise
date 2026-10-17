"""FastAPI demo endpoints. No auth, no scaling, no production infra (design doc
premise 4) — a local demo for interview purposes.

Two models, each switched on by an env var, at least one required:
- ADAPTER_DIR: the Phase 0 Banking77 classifier at POST /predict.
- POINTWISE_WEIGHTS (+ POINTWISE_METRICS for T): the Pointwise decision model at POST /v1/systemone
  (Phase 6), with a state-prefix cache. Optional: POINTWISE_DEVICE, POINTWISE_MERGE_LORA=1,
  POINTWISE_CACHE_ENTRIES, POINTWISE_CACHE_TOKENS, and the request limits POINTWISE_MAX_STATE_TOKENS (default 4096),
  POINTWISE_MAX_REQUEST_TOKENS (8192, state plus every question) and POINTWISE_MAX_QUESTIONS (32).
  POINTWISE_MAX_BODY_BYTES (1 MiB; over it a 413 before any parsing). Overload: POINTWISE_MAX_QUEUE (16 requests
  waiting for the GPU, 0 = never wait) and POINTWISE_QUEUE_TIMEOUT_S (10); past either the server answers 503 with
  Retry-After. A route whose model is not configured also answers 503.

Applies: fail-fast on missing/corrupted adapter (1A), structured error
responses (2A), input validation with length cap (3A), metadata-only logging
that never persists raw input text (3B).
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_task_config
from src.evaluate import generate_labels_batch
from src.evaluate_core import format_label_for_display, parse_label
from src.pointwise.encoding import ContextOverflow, UnsafeText
from src.pointwise.packing import StateCache
from src.pointwise.schema import SystemOneRequest
from src.train import MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH = 500

_config = None
_model = None
_tokenizer = None
_pointwise = None  # DecisionModel, set by load_pointwise
_state_cache: StateCache | None = None
_gpu_lock = threading.Lock()  # the StateCache and the GPU serve one request at a time, for both routes
# Request limits (set by load_pointwise). Defaults are conservative: run 3 was trained on states of at most 384
# tokens and its accuracy beyond that was never measured (docs/production.md).
DEFAULT_MAX_STATE_TOKENS, DEFAULT_MAX_REQUEST_TOKENS, DEFAULT_MAX_QUESTIONS = 4096, 8192, 32
DEFAULT_MAX_BODY_BYTES = 1 << 20  # 1 MiB: far above 8,192 tokens of text, far below what could exhaust memory
DEFAULT_CACHE_ENTRIES, DEFAULT_CACHE_TOKENS = 8, 16384
DEFAULT_MAX_QUEUE, DEFAULT_QUEUE_TIMEOUT_S = 16, 10.0
# Every waiting request holds one thread of FastAPI's sync-route pool (anyio default 40), as does the one holding the GPU
# and each request still tokenising. A longer queue would fill the pool, and the requests beyond it would wait unseen,
# outside the queue and its timeout. 24 leaves some headroom for the tokenising threads. Any health or metrics route
# added later must be `async def`, so it never needs a pool thread.
MAX_QUEUE_LIMIT = 24
_max_questions = DEFAULT_MAX_QUESTIONS
_max_body_bytes = DEFAULT_MAX_BODY_BYTES
# Overload: a request that finds the GPU busy may wait only if fewer than _max_queue are already waiting, and for at
# most _queue_timeout seconds; otherwise the server answers 503 + Retry-After at once instead of letting latency
# grow without bound. _max_queue = 0 means "never wait, reject if busy".
_max_queue = DEFAULT_MAX_QUEUE
_queue_timeout = DEFAULT_QUEUE_TIMEOUT_S
_waiting = 0
_rejected = 0  # requests turned away with a 503 for overload since start
_busy_since: float | None = None  # monotonic start of the request holding the GPU; a hung one shows as a growing age
_queue_lock = threading.Lock()


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=_MAX_MESSAGE_LENGTH)


class PredictResponse(BaseModel):
    predicted_label: str
    predicted_label_display: str


def load_model(adapter_dir: str) -> None:
    """1A: fails fast and refuses to start if the adapter checkpoint is
    missing/corrupted — never silently falls back to serving the base model,
    which would look like a working demo while actually showing unmodified
    base-model behavior."""
    global _config, _model, _tokenizer

    adapter_path = Path(adapter_dir)
    if not adapter_path.exists() or not any(adapter_path.iterdir()):
        logger.error(f"Adapter checkpoint not found or empty at {adapter_dir}")
        sys.exit(
            f"FATAL: adapter checkpoint missing or empty at '{adapter_dir}'. "
            f"Refusing to start rather than silently serving the base model. "
            f"Download the checkpoint from Drive before starting serve.py."
        )

    _config = load_task_config()
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

    try:
        _model = PeftModel.from_pretrained(base_model, adapter_dir)
    except Exception as exc:
        logger.error(f"Failed to load adapter from {adapter_dir}: {exc}")
        sys.exit(f"FATAL: adapter at '{adapter_dir}' failed to load: {exc}")

    logger.info(f"Model loaded successfully from adapter at {adapter_dir}")


def _overloaded(reason: str) -> HTTPException:
    """Count and log a turned-away request (metadata only), and build its 503."""
    global _rejected
    with _queue_lock:
        _rejected += 1
        waiting, since = _waiting, _busy_since  # one read each: the holder may clear _busy_since at any moment
    busy_for = None if since is None else time.monotonic() - since
    logger.warning(f"overloaded: {reason}; waiting={waiting} gpu_busy_for={'idle' if busy_for is None else f'{busy_for:.1f}s'}")
    return HTTPException(status_code=503, detail={"error": "overloaded", "message": reason}, headers={"Retry-After": "1"})


@contextmanager
def _gpu():
    """Hold the GPU lock for one request. If it is busy, wait only when the queue has room and only for
    _queue_timeout seconds; otherwise raise a 503. Only requests that really have to wait are counted as waiting."""
    global _waiting, _busy_since
    if not _gpu_lock.acquire(blocking=False):
        with _queue_lock:
            full = _waiting >= _max_queue
            _waiting += not full
        if full:
            raise _overloaded("queue is full")
        try:
            acquired = _gpu_lock.acquire(timeout=_queue_timeout)
        finally:
            with _queue_lock:
                _waiting -= 1
        if not acquired:
            raise _overloaded(f"waited {_queue_timeout:g}s for the GPU")
    with _queue_lock:
        _busy_since = time.monotonic()
    try:
        yield
    finally:
        with _queue_lock:
            _busy_since = None
        _gpu_lock.release()


class BodyLimit:
    """Rejects a request body over _max_body_bytes with a 413 before it is parsed or tokenised, whether the size comes
    from Content-Length or from a chunked stream. Read at request time, so a server can set the limit at startup."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = _max_body_bytes
        detail = {"error": "body_too_large", "message": f"request body is over the {limit:,}-byte limit"}
        declared = dict(scope["headers"]).get(b"content-length", b"0")
        if declared.isdigit() and int(declared) > limit:
            # Connection: close, because the declared body is not read and the connection cannot be reused
            return await JSONResponse({"detail": detail}, status_code=413, headers={"Connection": "close"})(scope, receive, send)
        total = 0

        async def limited_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b"")) if message["type"] == "http.request" else 0
            if total > limit:  # an HTTPException, which FastAPI's body parsing re-raises instead of turning into a 400
                raise HTTPException(status_code=413, detail=detail)
            return message

        await self.app(scope, limited_receive, send)


def load_pointwise(weights: str, metrics: str, device: str | None = None, merge_lora: bool = False,
                   cache_entries: int = DEFAULT_CACHE_ENTRIES, cache_tokens: int = DEFAULT_CACHE_TOKENS,
                   max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS, max_request_tokens: int = DEFAULT_MAX_REQUEST_TOKENS,
                   max_questions: int = DEFAULT_MAX_QUESTIONS) -> None:
    """Fails at startup, never on the first request: missing or unreadable weights or metrics, or a
    non-positive temperature, stop the server. T comes from run 3's metrics.json, fitted on dev."""
    global _pointwise, _state_cache, _max_questions
    from src.pointwise import train as T

    for name, path in (("POINTWISE_WEIGHTS", weights), ("POINTWISE_METRICS", metrics)):
        if not path or not Path(path).is_file():
            sys.exit(f"FATAL: {name} must point to an existing file, got {path!r}.")
    try:
        saved = json.loads(Path(metrics).read_text())
        temperature = float(saved["temperature"])
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state["lora"], state["head"]  # noqa: B018  the keys load_trainable reads
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"FATAL: could not read Pointwise weights {weights!r} or metrics {metrics!r}: {exc!r}")
    if "best_step" in saved and saved["best_step"] != state.get("step"):  # T was fitted on other weights
        sys.exit(f"FATAL: {metrics!r} is for step {saved['best_step']} but {weights!r} is step {state.get('step')}.")
    limits = {"cache_entries": cache_entries, "cache_tokens": cache_tokens, "max_state_tokens": max_state_tokens,
              "max_request_tokens": max_request_tokens, "max_questions": max_questions}
    if min(limits.values()) < 1:
        sys.exit(f"FATAL: cache sizes and request limits must be positive, got {limits}.")
    if max_request_tokens < max_state_tokens:
        sys.exit(f"FATAL: POINTWISE_MAX_REQUEST_TOKENS ({max_request_tokens}) is below POINTWISE_MAX_STATE_TOKENS ({max_state_tokens}).")
    if not (math.isfinite(temperature) and temperature > 0):
        sys.exit(f"FATAL: temperature in {metrics!r} must be a positive number, got {temperature}.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    targs = T.parse_args(["--data-dir", ".", "--output-dir", "."])  # run 3's model settings (LoRA r=16)
    model = T.build_model(targs, torch.device(device))
    T.load_trainable(model, state)
    if state["lora"] and not any(p.abs().sum() > 0 for n, p in model.backbone.named_parameters() if "lora_B" in n):
        sys.exit(f"FATAL: {weights!r} has adapter weights but none loaded into the model (adapter keys do not match).")
    if merge_lora:  # fold the adapter into the weights: one fewer matmul per layer, no more LoRA
        model.backbone = model.backbone.merge_and_unload()
    model.head.temperature = temperature
    model.max_state_tokens, model.max_request_tokens, _max_questions = max_state_tokens, max_request_tokens, max_questions
    _pointwise = model.eval()
    _state_cache = StateCache(cache_entries, cache_tokens)
    logger.info(f"Pointwise loaded on {device}: step {state.get('step')}, T={temperature:.3f}, merged={merge_lora}"
                + (" (T was fitted on the unmerged model; see scripts/phase6_latency.py for the drift)" if merge_lora else ""))


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        sys.exit(f"FATAL: {name} must be a number, got {os.environ[name]!r}.")
    if not 0 < value <= threading.TIMEOUT_MAX:  # also rejects inf and nan; acquire(timeout=inf) raises on every request
        sys.exit(f"FATAL: {name} must be a positive number up to {threading.TIMEOUT_MAX:g}, got {value}.")
    return value


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        sys.exit(f"FATAL: {name} must be 0 or 1, got {value!r}.")
    return value == "1"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        sys.exit(f"FATAL: {name} must be an integer, got {os.environ[name]!r}.")


def configure_from_env() -> None:
    global _max_queue, _queue_timeout, _max_body_bytes
    _max_queue = _env_int("POINTWISE_MAX_QUEUE", DEFAULT_MAX_QUEUE)
    _queue_timeout = _env_float("POINTWISE_QUEUE_TIMEOUT_S", DEFAULT_QUEUE_TIMEOUT_S)
    _max_body_bytes = _env_int("POINTWISE_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES)
    if not 0 <= _max_queue <= MAX_QUEUE_LIMIT:
        sys.exit(f"FATAL: POINTWISE_MAX_QUEUE must be between 0 and {MAX_QUEUE_LIMIT}, got {_max_queue}.")
    if _max_body_bytes < 1:
        sys.exit(f"FATAL: POINTWISE_MAX_BODY_BYTES must be positive, got {_max_body_bytes}.")
    adapter_dir, weights = os.environ.get("ADAPTER_DIR"), os.environ.get("POINTWISE_WEIGHTS")
    if not adapter_dir and not weights:
        sys.exit("FATAL: set ADAPTER_DIR (Phase 0 classifier) and/or POINTWISE_WEIGHTS and POINTWISE_METRICS (Pointwise).")
    if adapter_dir:
        load_model(adapter_dir)
    if weights:
        load_pointwise(
            weights,
            os.environ.get("POINTWISE_METRICS", ""),
            os.environ.get("POINTWISE_DEVICE"),
            _env_flag("POINTWISE_MERGE_LORA"),
            _env_int("POINTWISE_CACHE_ENTRIES", DEFAULT_CACHE_ENTRIES),
            _env_int("POINTWISE_CACHE_TOKENS", DEFAULT_CACHE_TOKENS),
            _env_int("POINTWISE_MAX_STATE_TOKENS", DEFAULT_MAX_STATE_TOKENS),
            _env_int("POINTWISE_MAX_REQUEST_TOKENS", DEFAULT_MAX_REQUEST_TOKENS),
            _env_int("POINTWISE_MAX_QUESTIONS", DEFAULT_MAX_QUESTIONS),
        )


@asynccontextmanager
async def _lifespan(_: FastAPI):
    configure_from_env()
    yield


app = FastAPI(title="Pointwise and Banking Support Intent Classifier", lifespan=_lifespan)
app.add_middleware(BodyLimit)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail={"error": "classifier_not_configured"})
    start = time.monotonic()
    # 3B: log request metadata only — never the raw input text, since a live
    # demo could have real-looking sensitive text typed into it.
    logger.info(f"predict request received, length={len(request.message)}")

    try:
        # Demo keeps the full label list in the prompt (D7 reversed) —
        # matches the fine-tuning distribution exactly, per config.format_prompt.
        prompt = _config.format_prompt(request.message)
        with _gpu():
            generated_texts = generate_labels_batch(_model, _tokenizer, [prompt], _config)
        predicted_label = parse_label(generated_texts[0], _config.labels)
    except HTTPException:  # e.g. the 503 from the queue: not a generation failure
        raise
    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.error(f"predict generation failed after {elapsed:.2f}s: {exc}")
        raise HTTPException(status_code=500, detail={"error": "generation_failed"}) from exc

    if predicted_label is None:
        elapsed = time.monotonic() - start
        logger.info(f"predict unresolved after {elapsed:.2f}s")
        raise HTTPException(status_code=500, detail={"error": "unresolved_prediction"})

    elapsed = time.monotonic() - start
    logger.info(f"predict succeeded in {elapsed:.2f}s")
    return PredictResponse(
        predicted_label=predicted_label,
        predicted_label_display=format_label_for_display(predicted_label),
    )


@app.post("/v1/systemone")
def systemone(request: SystemOneRequest) -> dict:
    """TypeSafe's /v1/systemone contract (src/pointwise/schema.py). The body is validated by the schema's
    pydantic models, so a malformed request is a 422 before the model runs. Answers always name a declared option."""
    if _pointwise is None:
        raise HTTPException(status_code=503, detail={"error": "pointwise_not_configured"})
    if len(request.questions) > _max_questions:
        raise HTTPException(status_code=422, detail={
            "error": "too_many_questions", "message": f"{len(request.questions)} questions, over the limit of {_max_questions}"})
    start = time.monotonic()
    try:
        # CPU-bound, and rejects oversize or invalid text, before taking the GPU. No lock: request threads share one
        # tokenizer, which a stress test shows is safe (test_the_shared_tokenizer_encodes_identically_from_many_threads).
        encoded = _pointwise.encode(request)
        with _gpu():
            hits = _state_cache.hits
            response = _pointwise.decide(request, state_cache=_state_cache, encoded=encoded)
            cached = _state_cache.hits > hits
    except ContextOverflow as exc:
        raise HTTPException(status_code=422, detail={"error": "context_overflow", "message": str(exc)}) from exc
    except UnsafeText as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_request", "message": str(exc)}) from exc
    except HTTPException:  # e.g. the 503 from the queue: not a decision failure
        raise
    except Exception as exc:
        logger.error(f"systemone failed after {time.monotonic() - start:.2f}s: {type(exc).__name__}")  # 3B: no exception text
        raise HTTPException(status_code=500, detail={"error": "decision_failed"}) from exc
    # 3B: metadata only, never the state text.
    logger.info(f"systemone questions={len(request.questions)} tokens={response['usage']['input_tokens']} "
                f"state_cache={'hit' if cached else 'miss'} {1000 * (time.monotonic() - start):.0f}ms")
    return response


# Note: a /health + /model-info endpoint was explicitly deferred to TODOS.md
# (item 6, from the office-hours cherry-pick ceremony) — not built here.
