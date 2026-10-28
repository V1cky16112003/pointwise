"""Phase 6: the state-prefix cache and POST /v1/systemone, on a tiny random model (no weights downloaded).

Skipped where torch/fastapi are absent (CI) or the Qwen tokenizer isn't cached. The checks that matter:
a cached request gives the same probabilities as an uncached one, two states never share a cache entry,
the endpoint's body is exactly model.decide()'s, and bad input is a 4xx, never a 500.
"""
import json
import logging
import re
import threading
import time

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("peft")  # src.serve imports it at the top

from fastapi.testclient import TestClient  # noqa: E402

from src import serve  # noqa: E402
from src.pointwise.encoding import UnsafeText  # noqa: E402
from src.pointwise.model import DecisionModel  # noqa: E402
from src.pointwise.packing import StateCache  # noqa: E402
from src.pointwise.schema import SystemOneRequest  # noqa: E402

# fp32 on CPU: cached and uncached runs are different matmul shapes, so they differ by float noise only.
FP32_TOL = 1e-5
# fp16 (the T4's dtype) keeps ~3 significant digits, and the cached and packed passes add up in a different order.
FP16_TOL = 5e-3

REQUEST = {
    "state": "Shoes arrived two weeks late and in the wrong size. Also two charges on my card.",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"returns": "Refunds, wrong items", "shipping": None, "billing": "Charges"}},
        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
    },
}
OTHER_STATE = {**REQUEST, "state": "I would like to change the delivery address of my order before it ships."}


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached locally")


def tiny_backbone(tokenizer):
    torch.manual_seed(0)
    config = transformers.Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                      head_dim=16, attn_implementation="sdpa")
    return transformers.Qwen3Model(config).eval()


@pytest.fixture(scope="module")
def model(tokenizer):
    return DecisionModel(tiny_backbone(tokenizer), tokenizer).eval()


@pytest.fixture(scope="module")
def lora_model(tokenizer):
    """A LoRA-wrapped backbone with a non-zero adapter, so merging changes weights and caching runs the peft path."""
    peft = pytest.importorskip("peft")
    backbone = peft.get_peft_model(
        tiny_backbone(tokenizer),
        peft.LoraConfig(task_type="FEATURE_EXTRACTION", r=4, target_modules=["q_proj", "v_proj", "up_proj"]),
    )
    with torch.no_grad():
        for name, p in backbone.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.05)
    return DecisionModel(backbone.eval(), tokenizer).eval()


def probs(model, data, cache=None):
    return model.probs(model.encode(SystemOneRequest.model_validate(data)), cache)


def assert_close(a, b, tol):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x == pytest.approx(y, abs=tol)


def flat(p):
    return [v for question in p for v in question]


def test_cached_probabilities_equal_uncached_on_a_miss_and_on_a_hit(model):
    cache = StateCache()
    plain = flat(probs(model, REQUEST))
    assert_close(flat(probs(model, REQUEST, cache)), plain, FP32_TOL)  # miss: prefill + branch
    assert_close(flat(probs(model, REQUEST, cache)), plain, FP32_TOL)  # hit: branch only
    assert (cache.hits, cache.misses, len(cache)) == (1, 1, 1)


def test_cached_probabilities_equal_uncached_in_fp16(tokenizer):
    half = DecisionModel(tiny_backbone(tokenizer).half(), tokenizer).eval()
    cache = StateCache()
    plain = flat(probs(half, REQUEST))
    probs(half, REQUEST, cache)
    assert_close(flat(probs(half, REQUEST, cache)), plain, FP16_TOL)


def test_a_request_with_fewer_questions_after_a_larger_one_is_unaffected(model):
    """branch_features crops the stored cache back to the state; a leftover question would corrupt the next hit."""
    cache = StateCache()
    probs(model, REQUEST, cache)
    one = {**REQUEST, "questions": {"escalate": REQUEST["questions"]["escalate"]}}
    assert_close(flat(probs(model, one, cache)), flat(probs(model, one)), FP32_TOL)
    assert_close(flat(probs(model, REQUEST, cache)), flat(probs(model, REQUEST)), FP32_TOL)


def test_two_different_states_never_share_an_entry(model):
    cache = StateCache()
    assert StateCache.key([1, 2, 3]) != StateCache.key([1, 2, 4]) != StateCache.key([1, 2])
    a, b = flat(probs(model, REQUEST)), flat(probs(model, OTHER_STATE))
    assert a != b
    for data, want in ((REQUEST, a), (OTHER_STATE, b), (REQUEST, a), (OTHER_STATE, b)):  # interleaved
        assert_close(flat(probs(model, data, cache)), want, FP32_TOL)
    assert (len(cache), cache.misses, cache.hits) == (2, 2, 2)


def test_stored_entries_hold_only_their_state_not_the_cropped_question_rows(model):
    cache = StateCache()
    probs(model, REQUEST, cache)
    probs(model, REQUEST, cache)  # a hit grows the tensors again before cropping
    (entry, n), = cache._entries.values()
    cfg = model.backbone.config
    per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 4  # keys+values, fp32
    assert cache.nbytes() == n * per_token
    for layer in entry.layers:
        assert layer.keys.untyped_storage().nbytes() == layer.keys.numel() * layer.keys.element_size()


def test_least_recently_used_state_is_evicted_first(model):
    cache = StateCache(max_entries=2)
    states = [{**REQUEST, "state": f"Order number {n} arrived broken."} for n in range(3)]
    for s in states:
        probs(model, s, cache)
    assert len(cache) == 2
    probs(model, states[1], cache)  # touch 1: now 2 is older than 1
    assert cache.hits == 1
    probs(model, {**REQUEST, "state": "A fourth, new state."}, cache)  # evicts 2, not 1
    probs(model, states[1], cache)
    assert cache.hits == 2
    probs(model, states[2], cache)
    assert cache.hits == 2  # 2 was evicted, so this was a miss


def test_token_budget_bounds_the_cache_and_an_oversized_state_is_not_stored(model):
    na = len(model.encode(SystemOneRequest.model_validate(REQUEST)).state_ids)
    nb = len(model.encode(SystemOneRequest.model_validate(OTHER_STATE)).state_ids)
    cache = StateCache(max_entries=10, max_tokens=max(na, nb))  # each state fits alone, the two together do not
    probs(model, REQUEST, cache)
    probs(model, OTHER_STATE, cache)
    assert len(cache) == 1 and cache.hits == 0
    probs(model, OTHER_STATE, cache)  # the newer one survived
    assert cache.hits == 1
    probs(model, REQUEST, cache)  # the older one was evicted for the budget (not the entry count)
    assert cache.hits == 1
    tiny = StateCache(max_tokens=min(na, nb) - 1)
    assert_close(flat(probs(model, REQUEST, tiny)), flat(probs(model, REQUEST)), FP32_TOL)  # still answered, one packed pass
    assert len(tiny) == 0 and tiny.misses == 1


def test_a_failure_inside_the_branch_pass_still_leaves_a_state_only_entry(model, monkeypatch):
    from src.pointwise import packing

    cache = StateCache()
    probs(model, REQUEST, cache)
    real = packing.branch_features

    def grow_then_fail(m, c, packed):  # the real pass grows the cache and crops it, then the request fails
        real(m, c, packed)
        raise RuntimeError("out of memory")

    monkeypatch.setattr(packing, "branch_features", grow_then_fail)
    with pytest.raises(RuntimeError):
        probs(model, REQUEST, cache)
    (entry, n), = cache._entries.values()
    assert all(layer.keys.untyped_storage().nbytes() == layer.keys.numel() * layer.keys.element_size() for layer in entry.layers)


def test_cache_works_through_a_lora_wrapped_backbone_and_merging_keeps_the_answers(lora_model):
    peft = pytest.importorskip("peft")
    cache = StateCache()
    plain = flat(probs(lora_model, REQUEST))
    probs(lora_model, REQUEST, cache)
    assert_close(flat(probs(lora_model, REQUEST, cache)), plain, FP32_TOL)
    assert isinstance(lora_model.backbone, peft.PeftModel)
    lora_model.backbone = lora_model.backbone.merge_and_unload()  # last use of this module-scoped fixture
    assert not isinstance(lora_model.backbone, peft.PeftModel)
    assert_close(flat(probs(lora_model, REQUEST)), plain, 1e-4)


@pytest.fixture
def client(model, monkeypatch):
    monkeypatch.setattr(serve, "_pointwise", model)
    monkeypatch.setattr(serve, "_state_cache", StateCache())
    return TestClient(serve.app, raise_server_exceptions=False)  # no `with`: skip the startup that needs env vars


def test_endpoint_body_is_exactly_decide(client, model):
    for data in (REQUEST, OTHER_STATE):
        got = client.post("/v1/systemone", json=data)
        assert got.status_code == 200
        assert got.json() == json.loads(json.dumps(model.decide(data, StateCache())))  # the same path, exactly
        plain = json.loads(json.dumps(model.decide(data)))  # uncached: equal up to float noise, same answers
        for key, answer in got.json()["answers"].items():
            if "probabilities" in answer:
                assert answer.get("choice") == plain["answers"][key].get("choice")  # a score answer has none
                assert_close(list(answer["probabilities"].values()), list(plain["answers"][key]["probabilities"].values()), 1e-3)


def test_endpoint_answers_name_declared_options_and_cache_hits_do_not_change_them(client):
    first, second = (client.post("/v1/systemone", json=REQUEST).json() for _ in range(2))
    assert serve._state_cache.hits == 1
    assert first == second
    answers = first["answers"]
    options = set(REQUEST["questions"]["department"]["criteria"])
    assert answers["department"]["choice"] in options
    assert set(answers["department"]["probabilities"]) == options
    assert 0 <= answers["escalate"]["noul"] <= 1
    assert 0 <= answers["frustration"]["score"] <= 2
    assert first["usage"]["output_tokens"] == 0


@pytest.mark.parametrize("body", [
    {},
    {"state": "x"},
    {"state": "x", "questions": {}},
    {"state": "x", "questions": {"q": {"type": "pick", "instructions": "?"}}},
    {"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {}}}},
    {"state": "x", "questions": {"q": {"type": "score", "instructions": "?", "criteria": []}}},
    {"state": "x", "questions": {"q": {"type": "choice", "instructions": "?"}}},
])
def test_malformed_requests_are_a_422_not_a_500(client, body):
    assert client.post("/v1/systemone", json=body).status_code == 422


def test_non_json_body_is_a_4xx(client):
    got = client.post("/v1/systemone", content=b"not json", headers={"content-type": "application/json"})
    assert 400 <= got.status_code < 500


def test_a_state_over_the_token_limit_is_a_422(client):
    got = client.post("/v1/systemone", json={**REQUEST, "state": "hello " * 40000})
    assert got.status_code == 422 and got.json()["detail"]["error"] == "context_overflow"


def test_caller_text_the_encoder_refuses_is_a_422_but_other_value_errors_are_server_faults(client, monkeypatch):
    def refuse(*a, **k):
        raise UnsafeText("could not tokenise caller text")

    monkeypatch.setattr(serve._pointwise, "encode", refuse)
    got = client.post("/v1/systemone", json=REQUEST)
    assert got.status_code == 422 and got.json()["detail"]["error"] == "invalid_request"
    monkeypatch.setattr(serve._pointwise, "encode", lambda *a, **k: (_ for _ in ()).throw(ValueError("internal invariant")))
    got = client.post("/v1/systemone", json=REQUEST)
    assert got.status_code == 500 and got.json() == {"detail": {"error": "decision_failed"}}


def questions(n):
    return {f"q{i}": {"type": "noul", "instructions": f"Is point {i} true?"} for i in range(n)}


def test_too_many_questions_is_a_422_and_the_limit_itself_is_allowed(client, monkeypatch):
    monkeypatch.setattr(serve, "_max_questions", 3)
    assert client.post("/v1/systemone", json={**REQUEST, "questions": questions(3)}).status_code == 200
    got = client.post("/v1/systemone", json={**REQUEST, "questions": questions(4)})
    assert got.status_code == 422 and got.json()["detail"]["error"] == "too_many_questions"


def test_state_and_total_token_limits_are_422s_with_the_counts(client, monkeypatch):
    monkeypatch.setattr(serve._pointwise, "max_state_tokens", 40)
    monkeypatch.setattr(serve._pointwise, "max_request_tokens", 200)
    assert client.post("/v1/systemone", json={**REQUEST, "state": "short"}).status_code == 200
    long_state = client.post("/v1/systemone", json={**REQUEST, "state": "word " * 60})
    assert long_state.status_code == 422 and "over the 40-token limit" in long_state.json()["detail"]["message"]
    many = client.post("/v1/systemone", json={**REQUEST, "state": "short", "questions": questions(30)})
    assert many.status_code == 422 and "200-token limit" in many.json()["detail"]["message"]
    assert many.json()["detail"]["error"] == "context_overflow"


def test_loading_applies_the_limits_and_rejects_inconsistent_ones(tmp_path, tokenizer, monkeypatch):
    peft = pytest.importorskip("peft")
    from src.pointwise import train as T

    weights, metrics = write_state(tmp_path)
    fresh = DecisionModel(peft.get_peft_model(
        tiny_backbone(tokenizer), peft.LoraConfig(task_type="FEATURE_EXTRACTION", r=4, target_modules=["q_proj"])), tokenizer)
    monkeypatch.setattr(T, "build_model", lambda args, device: fresh)
    monkeypatch.setattr(T, "load_trainable", lambda model, state: None)
    monkeypatch.setattr(serve, "_pointwise", None)
    monkeypatch.setattr(serve, "_state_cache", None)
    monkeypatch.setattr(serve, "_max_questions", 32)
    with pytest.raises(SystemExit, match="below POINTWISE_MAX_STATE_TOKENS"):
        serve.load_pointwise(weights, metrics, device="cpu", max_state_tokens=500, max_request_tokens=100)
    with pytest.raises(SystemExit, match="must be positive"):
        serve.load_pointwise(weights, metrics, device="cpu", max_questions=0)
    serve.load_pointwise(weights, metrics, device="cpu", max_state_tokens=300, max_request_tokens=900, max_questions=5)
    assert (fresh.max_state_tokens, fresh.max_request_tokens, serve._max_questions) == (300, 900, 5)


def test_max_queue_zero_serves_an_idle_server_and_rejects_only_when_busy(client, monkeypatch):
    monkeypatch.setattr(serve, "_max_queue", 0)  # never wait
    assert client.post("/v1/systemone", json=REQUEST).status_code == 200  # idle: nothing to wait for
    with serve._gpu_lock:  # busy
        got = client.post("/v1/systemone", json=REQUEST)
    assert got.status_code == 503 and got.headers["retry-after"] == "1" and got.json()["detail"]["error"] == "overloaded"
    assert serve._waiting == 0
    assert client.post("/v1/systemone", json=REQUEST).status_code == 200


def test_a_request_that_waits_too_long_for_the_gpu_is_a_503_and_releases_its_place(client, monkeypatch):
    monkeypatch.setattr(serve, "_queue_timeout", 0.05)
    with serve._gpu_lock:  # another request holds the GPU
        got = client.post("/v1/systemone", json=REQUEST)
        assert got.status_code == 503 and "waited" in got.json()["detail"]["message"]
        assert serve._waiting == 0
    assert client.post("/v1/systemone", json=REQUEST).status_code == 200


def test_the_gpu_lock_is_released_after_a_failed_request(client, monkeypatch):
    monkeypatch.setattr(serve._pointwise, "decide", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert client.post("/v1/systemone", json=REQUEST).status_code == 500
    assert not serve._gpu_lock.locked()


def test_predict_shares_the_same_queue(monkeypatch):
    from src.config import load_task_config

    monkeypatch.setattr(serve, "_config", load_task_config())
    monkeypatch.setattr(serve, "_model", object())
    monkeypatch.setattr(serve, "_tokenizer", object())
    monkeypatch.setattr(serve, "_max_queue", 0)
    with serve._gpu_lock:
        got = TestClient(serve.app, raise_server_exceptions=False).post("/predict", json={"message": "hi"})
    assert got.status_code == 503 and got.json()["detail"]["error"] == "overloaded"


def test_bad_queue_settings_stop_startup(tmp_path, monkeypatch):
    weights, metrics = write_state(tmp_path)
    monkeypatch.delenv("ADAPTER_DIR", raising=False)
    monkeypatch.setenv("POINTWISE_WEIGHTS", weights)
    monkeypatch.setenv("POINTWISE_METRICS", metrics)
    monkeypatch.setattr(serve, "_max_queue", 16)  # restored afterwards: configure_from_env sets these globals
    monkeypatch.setattr(serve, "_queue_timeout", 10.0)
    monkeypatch.setenv("POINTWISE_QUEUE_TIMEOUT_S", "soon")
    with pytest.raises(SystemExit, match="must be a number"):
        serve.configure_from_env()
    for bad in ("0", "-1", "inf", "nan", "1e12"):  # inf and 1e12 would make acquire(timeout=...) raise on every request
        monkeypatch.setenv("POINTWISE_QUEUE_TIMEOUT_S", bad)
        with pytest.raises(SystemExit, match="must be a number from"):
            serve.configure_from_env()
    monkeypatch.setenv("POINTWISE_QUEUE_TIMEOUT_S", "5")
    monkeypatch.setenv("POINTWISE_MAX_QUEUE", "-1")
    with pytest.raises(SystemExit, match="must be between 0 and 24"):
        serve.configure_from_env()
    monkeypatch.setenv("POINTWISE_MAX_QUEUE", "25")
    with pytest.raises(SystemExit, match="must be between 0 and 24"):
        serve.configure_from_env()


def test_overload_is_logged_counted_and_shows_how_long_the_gpu_has_been_busy(client, monkeypatch, caplog):
    monkeypatch.setattr(serve, "_max_queue", 0)
    before = serve._rejected
    with caplog.at_level("WARNING", logger="src.serve"):
        with serve._gpu():  # a request holds the GPU
            assert serve._busy_since is not None
            assert client.post("/v1/systemone", json=REQUEST).status_code == 503
    assert serve._busy_since is None and serve._rejected == before + 1
    assert any("overloaded: queue is full" in r.message and "gpu_busy_for=" in r.message for r in caplog.records)
    assert "Shoes arrived" not in caplog.text  # metadata only


def test_gpu_gate_under_real_concurrency_keeps_one_holder_and_leaks_nothing(monkeypatch):
    import threading
    import time

    monkeypatch.setattr(serve, "_max_queue", 3)
    monkeypatch.setattr(serve, "_queue_timeout", 5.0)
    holders = peak = served = rejected = 0
    guard = threading.Lock()

    def worker():
        nonlocal holders, peak, served, rejected
        try:
            with serve._gpu():
                with guard:
                    holders += 1
                    peak = max(peak, holders)
                time.sleep(0.01)
                with guard:
                    holders -= 1
                    served += 1
        except serve.HTTPException as exc:
            assert exc.status_code == 503
            with guard:
                rejected += 1

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1 and served + rejected == 40 and served >= 1
    assert serve._waiting == 0 and not serve._gpu_lock.locked() and serve._busy_since is None


def test_oversize_bodies_are_a_413_before_parsing_whether_declared_or_chunked(client, monkeypatch):
    monkeypatch.setattr(serve, "_max_body_bytes", 300)
    small = client.post("/v1/systemone", json={"state": "x", "questions": questions(1)})
    assert small.status_code == 200
    big = {**REQUEST, "state": "word " * 400}
    declared = client.post("/v1/systemone", json=big)
    assert declared.status_code == 413 and declared.json()["detail"]["error"] == "body_too_large"
    assert declared.headers["connection"] == "close"

    def chunks():  # no Content-Length: sent as a chunked stream
        data = json.dumps(big).encode()
        for i in range(0, len(data), 100):
            yield data[i : i + 100]

    chunked = client.post("/v1/systemone", content=chunks(), headers={"content-type": "application/json"})
    assert chunked.status_code == 413 and chunked.json()["detail"]["error"] == "body_too_large"
    assert not serve._gpu_lock.locked() and serve._waiting == 0  # never reached the model


def test_encoding_stops_tokenising_as_soon_as_the_request_is_over_the_limit(tokenizer, monkeypatch):
    from src.pointwise import encoding
    from src.pointwise.encoding import ContextOverflow, encode_request

    calls = 0
    real = encoding.user_tokens

    def counting(tok, text):
        nonlocal calls
        calls += 1
        return real(tok, text)

    monkeypatch.setattr(encoding, "user_tokens", counting)
    request = SystemOneRequest.model_validate({**REQUEST, "state": "short", "questions": questions(200)})
    with pytest.raises(ContextOverflow, match="exceeds the 120-token limit"):
        encode_request(tokenizer, request, 4096, max_request_tokens=120)
    assert calls < 60  # three calls per question: all 200 questions would be 600 calls
    many_options = SystemOneRequest.model_validate({**REQUEST, "state": "short", "questions": {
        "q": {"type": "choice", "instructions": "Pick", "criteria": {f"option number {i}": None for i in range(255)}}}})
    calls = 0
    with pytest.raises(ContextOverflow):
        encode_request(tokenizer, many_options, 4096, max_request_tokens=100)
    assert calls < 40  # stopped inside the option list, not after all 255


def test_defaults_come_from_one_set_of_constants():
    import inspect

    sig = inspect.signature(serve.load_pointwise).parameters
    assert sig["max_state_tokens"].default == serve.DEFAULT_MAX_STATE_TOKENS
    assert sig["max_request_tokens"].default == serve.DEFAULT_MAX_REQUEST_TOKENS
    assert sig["max_questions"].default == serve.DEFAULT_MAX_QUESTIONS == serve._max_questions
    assert sig["cache_entries"].default == serve.DEFAULT_CACHE_ENTRIES and sig["cache_tokens"].default == serve.DEFAULT_CACHE_TOKENS


def test_overload_503_is_built_whether_or_not_a_request_holds_the_gpu(monkeypatch):
    monkeypatch.setattr(serve, "_busy_since", time.monotonic())
    assert serve._overloaded("queue is full").status_code == 503
    monkeypatch.setattr(serve, "_busy_since", None)
    assert serve._overloaded("queue is full").status_code == 503


def test_tokenising_and_rejection_happen_before_the_gpu_is_taken(client, monkeypatch):
    seen = []
    real = serve._pointwise.decide

    def spy(request, state_cache=None, encoded=None):
        assert encoded is not None  # encoded before the lock, passed in
        seen.append(serve._gpu_lock.locked())
        return real(request, state_cache=state_cache, encoded=encoded)

    monkeypatch.setattr(serve._pointwise, "decide", spy)
    assert client.post("/v1/systemone", json=REQUEST).status_code == 200 and seen == [True]  # decide runs under the lock
    monkeypatch.setattr(serve._pointwise, "max_state_tokens", 10)
    with serve._gpu_lock:  # the GPU is busy and the queue is closed...
        monkeypatch.setattr(serve, "_max_queue", 0)
        got = client.post("/v1/systemone", json={**REQUEST, "state": "word " * 50})
    assert got.status_code == 422  # ...yet the oversize request is told so (422), not turned away (503)


def test_the_request_limit_counts_the_decide_token(tokenizer):
    from src.pointwise.encoding import ContextOverflow, encode_request

    request = SystemOneRequest.model_validate({"state": "short", "questions": questions(1)})
    e = encode_request(tokenizer, request)
    exact = len(e.state_ids) + len(e.questions[0].ids)  # includes the decide token
    assert len(encode_request(tokenizer, request, 4096, exact).questions) == 1  # fits at exactly the limit
    with pytest.raises(ContextOverflow):
        encode_request(tokenizer, request, 4096, exact - 1)  # one token over, which only the decide token pushes over


def test_the_shared_tokenizer_encodes_identically_from_many_threads(tokenizer):
    """Request threads share one tokenizer, and the server encodes outside any lock. Many threads must get the same ids
    as a single thread, with no exception (HF tokenizers can raise 'Already borrowed' if misused across threads)."""
    import random
    import threading

    from src.pointwise.encoding import encode_request

    rng = random.Random(0)
    words = "refund order late broken card charge delivery wrong size invoice account login".split()
    requests = [SystemOneRequest.model_validate({"state": " ".join(rng.choice(words) for _ in range(rng.randint(5, 600))),
                                                 "questions": {"q": {"type": "choice", "instructions": "Which team?",
                                                                     "criteria": {w: f"about {w}" for w in words}}}})
                for _ in range(24)]
    expected = [encode_request(tokenizer, r, 10**6).state_ids for r in requests]
    bad = []

    def work(seed):
        local = random.Random(seed)
        for _ in range(25):
            i = local.randrange(len(requests))
            try:
                if encode_request(tokenizer, requests[i], 10**6).state_ids != expected[i]:
                    bad.append("mismatch")
            except Exception as exc:  # noqa: BLE001
                bad.append(repr(exc))

    threads = [threading.Thread(target=work, args=(k,)) for k in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert bad == []


def test_concurrent_http_requests_are_never_a_500(client):
    import threading

    results = []

    def call(i):
        results.append(client.post("/v1/systemone", json={**REQUEST, "state": f"Order {i} arrived late and broken."}).status_code)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(results) <= {200, 503} and results.count(200) >= 1
    assert serve._waiting == 0 and not serve._gpu_lock.locked()


@pytest.fixture
def ops(monkeypatch):
    """The server's shared operations state, with a fresh metrics store, restored afterwards."""
    from src.serve_ops import Metrics

    monkeypatch.setattr(serve.OPS, "metrics", Metrics())
    monkeypatch.setattr(serve.OPS, "api_key", None)
    monkeypatch.setattr(serve.OPS, "draining", False)
    return serve.OPS


def test_healthz_is_ok_and_readyz_is_ready_with_a_model(client, ops):
    assert client.get("/healthz").json() == {"status": "ok"}
    got = client.get("/readyz")
    assert got.status_code == 200 and got.json() == {"status": "ready"}


def test_readyz_is_503_and_logs_the_reason_once_when_not_ready(client, ops, monkeypatch, caplog):
    monkeypatch.setattr(serve, "_pointwise", None)
    monkeypatch.setattr(serve, "_model", None)
    monkeypatch.setattr(serve, "_last_not_ready", None)
    with caplog.at_level("WARNING", logger="src.serve"):
        for _ in range(3):
            got = client.get("/readyz")
            assert got.status_code == 503 and got.json() == {"status": "not ready"}  # no reason on an open endpoint
    assert caplog.text.count("readyz: not ready: no model loaded") == 1  # logged on change, not on every probe
    assert client.get("/healthz").status_code == 200  # alive, just not ready


def test_readyz_fails_when_one_request_has_held_the_gpu_too_long(client, ops, monkeypatch, caplog):
    monkeypatch.setattr(serve, "_stuck_after", 5.0)
    monkeypatch.setattr(serve, "_busy_since", time.monotonic() - 2)
    assert client.get("/readyz").status_code == 200  # busy, but not for long
    monkeypatch.setattr(serve, "_busy_since", time.monotonic() - 30)
    monkeypatch.setattr(serve, "_last_not_ready", None)
    with caplog.at_level("WARNING", logger="src.serve"):
        got = client.get("/readyz")
    assert got.status_code == 503 and got.json() == {"status": "not ready"}
    assert "held the GPU for 30s (limit 5s)" in caplog.text and "held the GPU" not in got.text  # timing stays in the log


def test_probe_and_metrics_routes_are_async_so_they_never_need_a_pool_thread():
    """The design rule behind MAX_QUEUE_LIMIT. This checks the rule, not pool starvation itself, which the test client
    cannot reproduce."""
    import inspect

    for route in (serve.healthz, serve.readyz, serve.metrics):
        assert inspect.iscoroutinefunction(route), route.__name__


def test_probes_answer_while_the_queue_is_full_and_the_gpu_is_held(client, ops, monkeypatch):
    monkeypatch.setattr(serve, "_max_queue", 0)
    with serve._gpu_lock:
        assert client.post("/v1/systemone", json=REQUEST).status_code == 503
        assert client.get("/healthz").status_code == 200 and client.get("/readyz").status_code == 200
        assert client.get("/metrics").status_code == 200


def test_metrics_report_requests_queue_gpu_and_cache(client, ops, monkeypatch):
    client.post("/v1/systemone", json=REQUEST)  # a miss
    client.post("/v1/systemone", json=REQUEST)  # a hit
    client.post("/v1/systemone", json={})  # a 422
    monkeypatch.setattr(serve, "_max_queue", 0)
    with serve._gpu_lock:
        client.post("/v1/systemone", json=REQUEST)  # a 503
    got = client.get("/metrics")
    assert got.status_code == 200 and got.headers["content-type"].startswith("text/plain; version=0.0.4")
    text = got.text
    assert 'pointwise_requests_total{route="/v1/systemone",status="200"} 2' in text
    assert 'pointwise_requests_total{route="/v1/systemone",status="422"} 1' in text
    assert 'pointwise_requests_total{route="/v1/systemone",status="503"} 1' in text
    assert "pointwise_state_cache_hits_total 1" in text and "pointwise_state_cache_misses_total 1" in text
    assert "pointwise_state_cache_entries 1" in text
    n = len(serve._pointwise.encode(SystemOneRequest.model_validate(REQUEST)).state_ids)
    assert f"pointwise_state_cache_tokens {n}" in text
    assert "pointwise_ready 1" in text and "pointwise_queue_waiting 0" in text and "pointwise_gpu_busy_seconds 0" in text
    assert re.search(r"pointwise_overload_rejections_total [1-9]", text)
    assert 'pointwise_request_duration_seconds_count{route="/v1/systemone"} 4' in text
    assert "Shoes arrived" not in text  # numbers only


def test_the_api_key_protects_the_real_routes_and_not_the_probes(client, ops):
    ops.api_key = "k-123"
    assert client.get("/healthz").status_code == 200 and client.get("/readyz").status_code == 200
    assert client.post("/v1/systemone", json=REQUEST).status_code == 401
    assert client.post("/predict", json={"message": "hi"}).status_code == 401
    assert client.get("/metrics").status_code == 401
    assert client.get("/openapi.json").status_code == 401  # the schema is not public either
    got = client.post("/v1/systemone", json=REQUEST, headers={"Authorization": "Bearer k-123"})
    assert got.status_code == 200 and "x-request-id" in got.headers


def test_a_bad_api_key_setting_stops_startup(tmp_path, monkeypatch):
    weights, metrics = write_state(tmp_path)
    monkeypatch.delenv("ADAPTER_DIR", raising=False)
    monkeypatch.setenv("POINTWISE_WEIGHTS", weights)
    monkeypatch.setenv("POINTWISE_METRICS", metrics)
    monkeypatch.setattr(serve.OPS, "api_key", None)
    for bad in ("", "   ", " padded ", "naïve-key"):
        monkeypatch.setenv("POINTWISE_API_KEY", bad)
        with pytest.raises(SystemExit, match="POINTWISE_API_KEY must be"):
            serve.configure_from_env()
    assert serve.OPS.api_key is None  # a rejected key is never half-applied
    monkeypatch.setenv("POINTWISE_API_KEY", "ok-key")
    monkeypatch.setenv("POINTWISE_STUCK_AFTER_S", "soon")
    with pytest.raises(SystemExit, match="POINTWISE_STUCK_AFTER_S must be a number"):
        serve.configure_from_env()


def test_the_key_and_timings_are_read_from_the_environment(tmp_path, monkeypatch):
    weights, metrics = write_state(tmp_path)
    monkeypatch.delenv("ADAPTER_DIR", raising=False)
    monkeypatch.setenv("POINTWISE_WEIGHTS", str(tmp_path / "missing.pt"))  # fails after the operations settings are read
    monkeypatch.setenv("POINTWISE_METRICS", metrics)
    monkeypatch.setenv("POINTWISE_API_KEY", "ok-key")
    monkeypatch.setenv("POINTWISE_STUCK_AFTER_S", "12")
    monkeypatch.setenv("POINTWISE_SHUTDOWN_GRACE_S", "7")
    for name, value in (("api_key", None), ("draining", False)):
        monkeypatch.setattr(serve.OPS, name, value)
    monkeypatch.setattr(serve, "_stuck_after", serve._stuck_after)
    monkeypatch.setattr(serve, "_shutdown_grace", serve._shutdown_grace)
    with pytest.raises(SystemExit, match="POINTWISE_WEIGHTS must point"):
        serve.configure_from_env()
    assert (serve.OPS.api_key, serve._stuck_after, serve._shutdown_grace) == ("ok-key", 12.0, 7.0)


def test_drain_waits_for_work_in_flight_then_returns(ops, monkeypatch):
    import asyncio

    monkeypatch.setattr(serve, "_shutdown_grace", 5.0)
    serve._gpu_lock.acquire()  # a request is running
    threading.Timer(0.2, serve._gpu_lock.release).start()
    started = time.monotonic()
    asyncio.run(serve._drain())
    assert ops.draining and 0.15 < time.monotonic() - started < 4  # waited for it, did not run to the deadline


def test_drain_gives_up_at_the_grace_period_and_says_so(ops, monkeypatch, caplog):
    import asyncio

    monkeypatch.setattr(serve, "_shutdown_grace", 30.0)
    monkeypatch.setattr(ops, "drain_wait", 0.2)  # what serve_main sets: its share, the rest having gone to uvicorn
    with serve._gpu_lock, caplog.at_level("WARNING", logger="src.serve"):
        started = time.monotonic()
        asyncio.run(serve._drain())
    assert "gave up waiting after 0.2s" in caplog.text and time.monotonic() - started < 1


def test_under_plain_uvicorn_the_drain_may_use_the_whole_grace(ops, monkeypatch, caplog):
    import asyncio

    monkeypatch.setattr(serve, "_shutdown_grace", 0.3)
    monkeypatch.setattr(ops, "drain_wait", None)  # nobody split the budget
    with serve._gpu_lock, caplog.at_level("WARNING", logger="src.serve"):
        asyncio.run(serve._drain())
    assert "gave up waiting after 0.3s" in caplog.text


def test_drain_also_waits_for_requests_that_have_not_reached_the_gpu_yet(ops, monkeypatch):
    import asyncio

    monkeypatch.setattr(serve, "_shutdown_grace", 5.0)
    monkeypatch.setattr(ops, "drain_wait", 1.0)
    ops.inflight = 1  # a request still tokenising: no GPU lock, not in the queue
    threading.Timer(0.2, lambda: setattr(ops, "inflight", 0)).start()
    started = time.monotonic()
    asyncio.run(serve._drain())
    assert 0.15 < time.monotonic() - started < 0.9 and ops.draining


def test_a_restarted_app_does_not_begin_life_draining(ops, model, monkeypatch):
    monkeypatch.setattr(serve, "configure_from_env", lambda: None)
    monkeypatch.setattr(serve, "_pointwise", model)
    monkeypatch.setattr(serve, "_shutdown_grace", 0.5)
    for _ in range(2):  # the first shutdown leaves the flag set; the second startup must clear it
        with TestClient(serve.app) as c:
            assert not ops.draining and c.get("/readyz").status_code == 200
        assert ops.draining


def test_a_draining_server_turns_data_requests_away_but_stays_observable(client, ops):
    ops.draining = True
    got = client.post("/v1/systemone", json=REQUEST)
    assert got.status_code == 503 and got.headers["retry-after"] == "1"
    ready = client.get("/readyz")
    assert ready.status_code == 503 and ready.json() == {"status": "not ready"}
    assert client.get("/healthz").status_code == 200 and client.get("/metrics").status_code == 200


@pytest.mark.parametrize("bad", ["abc", "nan", "inf", "-5", "0", "1", "0.5", "2.99"])
def test_the_runner_rejects_a_bad_grace_period_like_the_server_does(monkeypatch, bad):
    from src.serve_main import build_server

    monkeypatch.setenv("POINTWISE_SHUTDOWN_GRACE_S", bad)
    with pytest.raises(SystemExit, match="POINTWISE_SHUTDOWN_GRACE_S must be"):
        build_server()


@pytest.mark.parametrize("grace", [3.0, 3.5, 5.0, 12.0, 25.0, 29.9, 61.7])
def test_the_two_shutdown_waits_plus_the_margin_always_add_up_to_the_grace_period(grace):
    from src.serve_ops import SHUTDOWN_MARGIN_S, split_grace

    uvicorn_s, drain_s = split_grace(grace)
    assert isinstance(uvicorn_s, int) and uvicorn_s >= 1 and drain_s >= 0
    assert uvicorn_s + drain_s + SHUTDOWN_MARGIN_S == pytest.approx(grace)  # never over the budget an orchestrator allows


def test_the_default_grace_leaves_room_under_kubernetes_default_kill_timeout():
    from src.serve_ops import DEFAULT_SHUTDOWN_GRACE_S

    assert DEFAULT_SHUTDOWN_GRACE_S < 30 and serve.DEFAULT_SHUTDOWN_GRACE_S == DEFAULT_SHUTDOWN_GRACE_S


def test_a_shutdown_signal_that_arrives_before_the_app_starts_is_not_forgotten(ops, model, monkeypatch):
    import signal

    from src.serve_main import build_server

    monkeypatch.setattr(ops, "signalled", False)
    monkeypatch.setattr(serve, "configure_from_env", lambda: None)
    monkeypatch.setattr(serve, "_pointwise", model)
    monkeypatch.setattr(serve, "_shutdown_grace", 0.5)
    build_server().handle_exit(signal.SIGTERM, None)  # SIGTERM during the slow import and model load
    assert ops.signalled and ops.draining
    with TestClient(serve.app) as c:  # the lifespan then starts: it must not clear what the signal set
        assert ops.draining and c.get("/readyz").status_code == 503
    monkeypatch.setattr(ops, "signalled", False)


@pytest.mark.parametrize("bad", ["0", "-1", "65536", "70000", "x"])
def test_the_runner_rejects_a_bad_port(monkeypatch, bad):
    from src.serve_main import build_server

    monkeypatch.setenv("POINTWISE_PORT", bad)
    with pytest.raises(SystemExit, match="POINTWISE_PORT must be"):
        build_server()


def test_a_stuck_gpu_is_logged_once_not_once_per_probe_as_its_age_grows(client, ops, monkeypatch, caplog):
    monkeypatch.setattr(serve, "_stuck_after", 5.0)
    monkeypatch.setattr(serve, "_last_not_ready", None)
    with caplog.at_level("WARNING", logger="src.serve"):
        for age in (61, 71, 81, 9999):
            monkeypatch.setattr(serve, "_busy_since", time.monotonic() - age)
            assert client.get("/readyz").status_code == 503
    assert caplog.text.count("readyz: not ready") == 1


def test_extra_request_id_still_works_and_wins_over_the_current_one(caplog):
    """A global record factory would make `extra={"request_id": ...}` raise KeyError; a logger filter does not."""
    with caplog.at_level("INFO", logger="src.serve"):
        logging.getLogger("src.serve").info("caller-supplied", extra={"request_id": "from-caller"})
        logging.getLogger("src.serve").info("ambient")
    ids = {r.getMessage(): r.request_id for r in caplog.records}
    assert ids == {"caller-supplied": "from-caller", "ambient": "-"}
    logging.getLogger("some.other.library").warning("fine", extra={"request_id": "x"})  # unaffected anywhere in the process


def test_a_record_without_a_request_id_is_printed_with_a_dash_not_lost(capsys):
    """The formatter configuration serve.py applies to its root handlers, on a record that lacks the attribute."""
    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(serve._LOG_FORMAT, defaults={"request_id": "-"}))
    handler.handle(logging.makeLogRecord({"msg": "replayed from a queue", "levelname": "INFO", "levelno": 20}))
    assert "[-] replayed from a queue" in stream.getvalue() and "Logging error" not in capsys.readouterr().err


def test_a_new_run_starts_with_a_clean_readyz_log_state(ops, model, monkeypatch):
    monkeypatch.setattr(serve, "configure_from_env", lambda: None)
    monkeypatch.setattr(serve, "_pointwise", model)
    monkeypatch.setattr(serve, "_shutdown_grace", 0.5)
    monkeypatch.setattr(serve, "_last_not_ready", "held the GPU for Ns")  # left over from an earlier incident
    with TestClient(serve.app):
        assert serve._last_not_ready is None  # so a second incident is logged again


def test_the_lifespan_drains_on_shutdown(ops, monkeypatch):
    monkeypatch.setattr(serve, "configure_from_env", lambda: None)
    monkeypatch.setattr(serve, "_shutdown_grace", 1.0)
    with TestClient(serve.app) as c:  # startup and shutdown both run
        assert c.get("/healthz").status_code == 200 and not ops.draining
    assert ops.draining


def test_the_runner_starts_draining_at_the_shutdown_signal_before_uvicorn_stops(ops, monkeypatch):
    import signal

    from src.serve_main import build_server

    monkeypatch.setenv("POINTWISE_SHUTDOWN_GRACE_S", "12")
    monkeypatch.setattr(ops, "drain_wait", None)
    server = build_server()
    # 1 s held back, 80% of the other 11 s for uvicorn (8 s), and exactly the rest (3 s) for the server's own drain
    assert server.config.timeout_graceful_shutdown == 8 and ops.drain_wait == 3.0
    assert server.config.workers == 1 and not server.config.access_log
    assert not ops.draining and not server.should_exit
    server.handle_exit(signal.SIGTERM, None)
    assert ops.draining and ops.signalled and server.should_exit
    monkeypatch.setattr(ops, "signalled", False)


def test_a_bug_in_a_route_is_a_500_with_a_request_id_on_the_real_app(client, ops, monkeypatch, caplog):
    monkeypatch.setattr(serve, "readiness", lambda: (_ for _ in ()).throw(RuntimeError("secret internal detail")))
    with caplog.at_level("INFO", logger="src.serve"):
        got = client.get("/readyz", headers={"X-Request-Id": "find-me"})
    assert got.status_code == 500 and got.headers["x-request-id"] == "find-me"
    assert got.json() == {"detail": {"error": "internal_error"}} and "secret" not in got.text + caplog.text
    assert any(r.request_id == "find-me" and "unhandled RuntimeError" in r.getMessage() for r in caplog.records)


def test_an_unexpected_failure_is_a_generic_500_that_leaks_nothing(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(serve._pointwise, "decide", boom)
    got = client.post("/v1/systemone", json=REQUEST)
    assert got.status_code == 500 and got.json() == {"detail": {"error": "decision_failed"}}


def test_routes_answer_503_when_their_model_is_not_configured(monkeypatch):
    monkeypatch.setattr(serve, "_pointwise", None)
    monkeypatch.setattr(serve, "_model", None)
    c = TestClient(serve.app, raise_server_exceptions=False)
    assert c.post("/v1/systemone", json=REQUEST).status_code == 503
    assert c.post("/predict", json={"message": "hi"}).status_code == 503


def test_the_phase_0_predict_route_still_works(monkeypatch):
    from src.config import load_task_config

    config = load_task_config()
    label = config.labels[0]
    monkeypatch.setattr(serve, "_config", config)
    monkeypatch.setattr(serve, "_model", object())
    monkeypatch.setattr(serve, "_tokenizer", object())
    monkeypatch.setattr(serve, "generate_labels_batch", lambda *a, **k: [label])
    got = TestClient(serve.app).post("/predict", json={"message": "my card has not arrived"})
    assert got.status_code == 200 and got.json()["predicted_label"] == label


def write_state(tmp_path, temperature=1.5, lora=None):
    weights = tmp_path / "state.pt"
    torch.save({"step": 1541, "lora": lora or {}, "head": {}}, weights)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"temperature": temperature}))
    return str(weights), str(metrics)


@pytest.mark.parametrize("what", ["no_weights", "no_metrics", "bad_metrics", "zero_T", "negative_T", "bad_weights"])
def test_startup_fails_instead_of_serving_with_missing_or_bad_inputs(tmp_path, what):
    weights, metrics = write_state(tmp_path)
    if what == "no_weights":
        weights = str(tmp_path / "missing.pt")
    elif what == "no_metrics":
        metrics = ""
    elif what == "bad_metrics":
        (tmp_path / "metrics.json").write_text("{}")
    elif what in ("zero_T", "negative_T"):
        (tmp_path / "metrics.json").write_text(json.dumps({"temperature": 0 if what == "zero_T" else -1}))
    else:
        (tmp_path / "state.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(SystemExit, match="FATAL"):
        serve.load_pointwise(weights, metrics, device="cpu")


def test_metrics_from_other_weights_or_bad_cache_sizes_stop_startup(tmp_path, monkeypatch):
    weights, metrics = write_state(tmp_path)
    (tmp_path / "metrics.json").write_text(json.dumps({"temperature": 1.5, "best_step": 834}))  # weights are step 1541
    with pytest.raises(SystemExit, match="step 834"):
        serve.load_pointwise(weights, metrics, device="cpu")
    (tmp_path / "metrics.json").write_text(json.dumps({"temperature": 1.5}))
    with pytest.raises(SystemExit, match="positive"):
        serve.load_pointwise(weights, metrics, device="cpu", cache_entries=0)
    monkeypatch.delenv("ADAPTER_DIR", raising=False)  # a developer's shell may export it
    monkeypatch.setenv("POINTWISE_WEIGHTS", weights)
    monkeypatch.setenv("POINTWISE_METRICS", metrics)
    monkeypatch.setenv("POINTWISE_CACHE_ENTRIES", "many")
    with pytest.raises(SystemExit, match="must be an integer"):
        serve.configure_from_env()
    monkeypatch.setenv("POINTWISE_CACHE_ENTRIES", "4")
    monkeypatch.setenv("POINTWISE_MERGE_LORA", "true")
    with pytest.raises(SystemExit, match="must be 0 or 1"):
        serve.configure_from_env()


def test_startup_fails_when_adapter_weights_did_not_load(tmp_path, tokenizer, monkeypatch):
    peft = pytest.importorskip("peft")
    from src.pointwise import train as T

    weights, metrics = write_state(tmp_path, lora={"some.lora_B.weight": torch.ones(1)})
    fresh = DecisionModel(peft.get_peft_model(
        tiny_backbone(tokenizer), peft.LoraConfig(task_type="FEATURE_EXTRACTION", r=4, target_modules=["q_proj"])), tokenizer)
    monkeypatch.setattr(T, "build_model", lambda args, device: fresh)
    monkeypatch.setattr(T, "load_trainable", lambda model, state: None)  # keys that match nothing: lora_B stays zero
    with pytest.raises(SystemExit, match="none loaded"):
        serve.load_pointwise(weights, metrics, device="cpu")


def test_startup_needs_at_least_one_model(monkeypatch):
    for var in ("ADAPTER_DIR", "POINTWISE_WEIGHTS"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(SystemExit, match="FATAL"):
        serve.configure_from_env()


def test_loading_sets_the_temperature_from_metrics_and_can_merge_the_adapter(tmp_path, lora_model, tokenizer, monkeypatch):
    peft = pytest.importorskip("peft")
    from src.pointwise import train as T

    weights, metrics = write_state(tmp_path, temperature=1.516)
    fresh = DecisionModel(peft.get_peft_model(
        tiny_backbone(tokenizer),
        peft.LoraConfig(task_type="FEATURE_EXTRACTION", r=4, target_modules=["q_proj"])), tokenizer)
    monkeypatch.setattr(T, "build_model", lambda args, device: fresh)
    monkeypatch.setattr(T, "load_trainable", lambda model, state: None)
    monkeypatch.setattr(serve, "_pointwise", None)
    monkeypatch.setattr(serve, "_state_cache", None)  # restored afterwards, so no later test sees this one's cache
    serve.load_pointwise(weights, metrics, device="cpu", merge_lora=True, cache_entries=3, cache_tokens=100)
    assert serve._pointwise.head.temperature == pytest.approx(1.516)
    assert not isinstance(serve._pointwise.backbone, peft.PeftModel)
    assert (serve._state_cache.max_entries, serve._state_cache.max_tokens) == (3, 100)
    assert not serve._pointwise.training


def test_latency_measure_runs_every_configuration_and_merges_last(tokenizer):
    peft = pytest.importorskip("peft")
    from scripts import phase6_latency as L

    lora = DecisionModel(peft.get_peft_model(
        tiny_backbone(tokenizer),
        peft.LoraConfig(task_type="FEATURE_EXTRACTION", r=4, target_modules=["q_proj"])), tokenizer).eval()
    got = L.measure(lora, "Customer: my card has not arrived.\n" * 4, ks=(1, 2), reps=3, warmup=1)
    assert set(got["results"]) == {"a", "p5", "c", "d", "b", "e"} and got["merged"]
    assert all(cell["n"] == 3 and cell["median"] > 0 for row in got["results"].values() for cell in row.values())
    assert "| a. cold, unmerged |" in L.table(got)
    assert got["agreement"]["cache_miss_vs_uncached"]["max_prob_diff"] < 1e-4
    assert got["agreement"]["cache_hit_vs_uncached"]["max_prob_diff"] < 1e-4
    assert got["agreement"]["merged_vs_unmerged"]["questions"] == 3  # k=1 plus k=2
    assert got["cache_mb_per_state"] > 0


def test_gate_verdict_is_printed_for_k_8(tokenizer):
    from scripts import phase6_latency as L

    cell = lambda p95: {"n": 60, "median": p95 / 2, "mean": p95 / 2, "p95": p95, "max": p95}  # noqa: E731
    text = L.table({"results": {"a": {8: cell(200.0)}, "c": {8: cell(90.0)}}})
    assert "a. cold, unmerged: p95 200 ms at k=8, not met" in text and "c. cached state, unmerged: p95 90 ms at k=8, met" in text


def test_p95_is_the_95th_percentile_of_the_samples():
    from scripts import phase6_latency as L

    assert L.summarise([float(i) for i in range(1, 101)])["p95"] == 95.0
    assert L.summarise([3.0, 1.0, 2.0])["median"] == 2.0


def test_token_ratios_are_generate_and_parse_over_pointwise():
    from scripts.count_tokens import ratios

    got = ratios(state=500, questions=[20, 20, 20, 20], prompts=[520] * 4)
    assert (got["one_pass"], got["cached"], got["generate_and_parse"]) == (580, 80, 2080)
    assert got["x_one_pass"] == pytest.approx(2080 / 580) and got["x_cached"] == pytest.approx(26.0)


def test_a_miss_runs_one_backbone_pass_and_stores_only_the_state(model, monkeypatch):
    """Step 4 (b): a miss reads state + questions once, like a cold request, not prefill then questions."""
    calls = []
    forward = model.backbone.forward
    monkeypatch.setattr(model.backbone, "forward", lambda *a, **kw: calls.append(1) or forward(*a, **kw))
    cache = StateCache()
    encoded = model.encode(SystemOneRequest.model_validate(REQUEST))
    model.probs(encoded, cache)
    assert len(calls) == 1
    (stored, n), = cache._entries.values()
    assert n == len(encoded.state_ids) and all(layer.keys.shape[-2] == n for layer in stored.layers)


def test_adapter_is_merged_unless_switched_off(monkeypatch):
    from src.serve_ops import env_flag

    monkeypatch.delenv("POINTWISE_MERGE_LORA", raising=False)
    assert env_flag("POINTWISE_MERGE_LORA", default=True)
    monkeypatch.setenv("POINTWISE_MERGE_LORA", "0")
    assert not env_flag("POINTWISE_MERGE_LORA", default=True)
