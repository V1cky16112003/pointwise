"""Phase 6: the state-prefix cache and POST /v1/systemone, on a tiny random model (no weights downloaded).

Skipped where torch/fastapi are absent (CI) or the Qwen tokenizer isn't cached. The checks that matter:
a cached request gives the same probabilities as an uncached one, two states never share a cache entry,
the endpoint's body is exactly model.decide()'s, and bad input is a 4xx, never a 500.
"""
import json

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
