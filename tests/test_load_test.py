"""Step 7 load test: its statistics, request mix and answer comparison, and the driver against the tiny-model app."""
import asyncio
import copy

import pytest

httpx = pytest.importorskip("httpx")

from scripts import load_test as L  # noqa: E402


def test_percentile_is_nearest_rank():
    samples = list(range(1, 101))
    assert (L.percentile(samples, 50), L.percentile(samples, 95), L.percentile(samples, 99)) == (50, 95, 99)
    assert L.percentile([7.0], 99) == 7.0
    assert L.summarise([])["n"] == 0


def test_request_mix_repeats_only_recent_states_and_never_reuses_across_runs():
    ids = L.request_mix(1000, 0.5, seed=1)
    new = [i for n, i in enumerate(ids) if i not in ids[:n]]
    assert new == list(range(len(new)))  # new states are numbered in order of first use
    assert 0.4 < 1 - len(new) / len(ids) < 0.6
    for n, i in enumerate(ids):  # a repeat is one of the last RECENT new states at that point
        if i in ids[:n]:
            seen = [j for j in new if j in ids[:n]]
            assert i in seen[-L.RECENT:]
    assert L.request_mix(5, 0.0, seed=1, first_new=10) == [10, 11, 12, 13, 14]
    assert min(L.request_mix(50, 0.5, seed=2, first_new=max(ids) + 1)) > max(ids)


def test_states_are_unique_and_long_enough():
    count = lambda text: len(text.split())  # noqa: E731
    msgs = [f"message {i} about my card" for i in range(30)]
    a, b = L.make_state(0, msgs, count, 100), L.make_state(1, msgs, count, 100)
    assert a != b and count(a) >= 100


ANSWERS = {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.3, "c": 0.1}}}


def test_compare_answers_catches_another_states_answers_and_records_the_margin():
    same = L.compare_answers(copy.deepcopy(ANSWERS), ANSWERS)
    assert same == {"max_prob_diff": 0.0, "flips": [], "questions": 1}
    other = {"q": {"type": "choice", "choice": "b", "probabilities": {"a": 0.2, "b": 0.7, "c": 0.1}}}
    got = L.compare_answers(other, ANSWERS)
    assert got["max_prob_diff"] == pytest.approx(0.4) and got["flips"] == [{"question": "q", "margin": pytest.approx(0.3)}]


def test_metrics_parsing_and_deltas():
    text = ('# HELP x\npointwise_requests_total{route="/v1/systemone",status="200"} 5\n'
            'pointwise_state_cache_hits_total 3\npointwise_queue_waiting 2\n')
    before = L.parse_metrics(text)
    after = {**before, 'pointwise_requests_total{route="/v1/systemone",status="200"}': 9.0,
             'pointwise_requests_total{route="/v1/systemone",status="503"}': 1.0}
    assert L.metrics_delta(before, after) == {'pointwise_requests_total{route="/v1/systemone",status="200"}': 4.0,
                                              'pointwise_requests_total{route="/v1/systemone",status="503"}': 1.0,
                                              "pointwise_state_cache_hits_total": 0.0}


def test_level_row_counts_statuses_and_littles_law():
    records = [{"status": 200, "ms": 100.0}] * 8 + [{"status": 503, "ms": 1.0}, {"status": None, "error": "ConnectError", "ms": 1.0}]
    row = L.level_row(records, wall_s=0.2, clients=2)
    assert row["statuses"] == {"200": 8, "503": 1, "ConnectError": 1}
    assert row["throughput_rps"] == 40.0 and row["littles_law_in_flight"] == pytest.approx(4.0)


# The driver end to end against the real app on a tiny random model, so a parse bug is found here, not on Kaggle.
def test_driver_against_the_app_and_every_answer_matches_decide(monkeypatch):
    serving = pytest.importorskip("tests.test_pointwise_serving")
    from src import serve
    from src.pointwise.model import DecisionModel
    from src.pointwise.packing import StateCache

    try:
        tok = serving.transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached locally")
    model = DecisionModel(serving.tiny_backbone(tok), tok).eval()
    monkeypatch.setattr(serve, "_pointwise", model)
    monkeypatch.setattr(serve, "_state_cache", StateCache())
    msgs = [f"my card {i} was charged twice" for i in range(20)]
    bodies = {}
    body_for = lambda i: bodies.setdefault(i, L.payload(L.make_state(i, msgs, lambda t: len(tok.encode(t)), 60), k=3))  # noqa: E731

    async def go():
        transport = httpx.ASGITransport(app=serve.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await L.closed_loop(client, L.request_mix(12, 0.5, seed=0), body_for, 4, {})

    records, wall = asyncio.run(go())
    assert len(records) == 12 and all(r["status"] == 200 for r in records) and wall > 0
    for r in records:
        ref = model.decide(body_for(r["state"]), state_cache=StateCache())["answers"]
        assert L.compare_answers(r["answers"], ref)["max_prob_diff"] < 1e-3
    swapped = model.decide(body_for(records[0]["state"] + 1000), state_cache=StateCache())["answers"]
    assert L.compare_answers(records[0]["answers"], swapped)["max_prob_diff"] > 0  # a mixed-up state is visible
