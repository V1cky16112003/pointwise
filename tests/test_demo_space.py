"""The demo's logic (demo/pointwise_client.py) and its presets. No network. The presets are read from
demo/pointwise_app.py with ast, so gradio is not needed."""
import ast
import json
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
import pointwise_client as client  # noqa: E402

from src.pointwise.schema import SystemOneRequest  # noqa: E402

# A real response from the deployed service (docs/modal.md, 2026-11-03), with a noul and a made-up score added for shape.
RESP = {
    "model": "pointwise-latest",
    "answers": {
        "department": {"type": "choice", "choice": "returns", "confidence": 0.3219,
                       "probabilities": {"returns": 0.5479, "shipping": 0.1024, "billing": 0.3497}},
        "escalate": {"type": "noul", "noul": 0.5904},
        "stars": {"type": "score", "score": 1.2, "confidence": 0.5, "legend": {"0": "bad", "1": "ok"},
                  "probabilities": {"0": 0.2, "1": 0.8}},
    },
    "usage": {"input_tokens": 60, "output_tokens": 0},
}
QUESTIONS = '{"q": {"type": "noul", "instructions": "Is it?"}}'


def _presets():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "demo" / "pointwise_app.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "PRESETS")
    return ast.literal_eval(node.value)


@pytest.mark.parametrize("name, preset", _presets().items())
def test_every_preset_is_a_valid_request(name, preset):
    assert len(_presets()) == 3
    body, err = client.build_body(
        preset["state"] if isinstance(preset["state"], str) else json.dumps(preset["state"]),
        json.dumps(preset["questions"]),
    )
    assert err is None
    SystemOneRequest(**body)
    assert 1 <= len(preset["questions"]) <= client.MAX_QUESTIONS


@pytest.mark.parametrize("state, questions, word", [
    ("", QUESTIONS, "state"),
    ("x" * 3001, QUESTIONS, "too long"),
    ("hi", "{not json", "valid JSON"),
    ("hi", "[]", "object"),
    ("hi", "{}", "object"),
    ("hi", '{"a":{},"b":{},"c":{},"d":{},"e":{}}', "at most"),
    ("hi", '{"a":{"type":"choice","criteria":{%s}}}' % ",".join(f'"o{i}":null' for i in range(11)), "more than"),
])
def test_input_limits(state, questions, word):
    body, err = client.build_body(state, questions)
    assert body is None and word in err


def test_json_state_is_parsed_but_bad_json_stays_text():
    assert client.build_body('{"a": 1}', QUESTIONS)[0]["state"] == {"a": 1}
    assert client.build_body("{oops", QUESTIONS)[0]["state"] == "{oops"


def test_cards_show_every_answer_and_escape_html():
    out = client.cards(RESP, 500)
    assert "returns" in out and "p(yes) 0.59" in out and "1.20 of 1" in out and "60 input tokens" in out
    hostile = {"answers": {"<script>": {"type": "noul", "noul": 0.1}}, "usage": {}}
    assert "<script>" not in client.cards(hostile, 1)


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload

    def json(self):
        if self._payload is None:
            raise ValueError
        return self._payload


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("POINTWISE_URL", "https://example.test/")
    monkeypatch.setenv("POINTWISE_API_KEY", "k")


def test_call_not_configured(monkeypatch):
    monkeypatch.delenv("POINTWISE_URL", raising=False)
    assert "not configured" in client.call({})[1]


def test_call_success_sends_key_and_hits_the_right_url(configured, monkeypatch):
    seen = {}

    def post(url, json, timeout, headers):
        seen.update(url=url, auth=headers["Authorization"], timeout=timeout)
        return FakeResponse(200, RESP)

    monkeypatch.setattr(client.requests, "post", post)
    assert client.call({"state": "x"}) == (RESP, None)
    assert seen == {"url": "https://example.test/v1/systemone", "auth": "Bearer k", "timeout": client.TIMEOUT_S}


@pytest.mark.parametrize("outcome, word", [
    (FakeResponse(401), "key"),
    (FakeResponse(503), "busy"),
    (FakeResponse(422), "rejected"),
    (FakeResponse(500), "500"),
    (FakeResponse(200, None), "could not read"),
    (requests.Timeout(), "did not answer"),
    (requests.ConnectionError(), "Could not reach"),
])
def test_call_errors_become_messages(configured, monkeypatch, outcome, word):
    def post(*a, **k):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(client.requests, "post", post)
    resp, err = client.call({})
    assert resp is None and word in err
