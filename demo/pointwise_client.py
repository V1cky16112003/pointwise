"""Everything the demo (demo/pointwise_app.py) does except the screen: check the input, call the service, turn the answer
into cards. No gradio import, so tests/test_demo_space.py runs without it."""
import html
import json
import os

import requests

TIMEOUT_S = 90  # a cold GPU takes about 20 s (docs/modal.md); this leaves room for a slow start
MAX_STATE_CHARS = 3000  # the model was trained on states up to 384 tokens (docs/production.md); this is already generous
MAX_QUESTIONS = 4
MAX_OPTIONS = 10

ERRORS = {
    401: "The demo is misconfigured (the service rejected its key).",
    413: "That request is too large.",
    422: "The service rejected this request. Check that the questions are valid.",
    503: "The service is busy. Try again in a few seconds.",
}


def parse_state(text):
    """Plain text, or a JSON object or array."""
    t = (text or "").strip()
    if t[:1] in "{[":
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
    return t


def build_body(state_text, questions_text):
    """(request body, None), or (None, error message). Checks the limits that keep one click cheap."""
    state = parse_state(state_text)
    if not state:
        return None, "Write some text in the state box."
    if len(json.dumps(state)) > MAX_STATE_CHARS:
        return None, f"The state is too long for this demo (limit {MAX_STATE_CHARS} characters)."
    try:
        questions = json.loads(questions_text or "")
    except json.JSONDecodeError as e:
        return None, f"The questions are not valid JSON: {e}"
    if not isinstance(questions, dict) or not questions:
        return None, "The questions must be a JSON object with at least one question."
    if len(questions) > MAX_QUESTIONS:
        return None, f"This demo allows at most {MAX_QUESTIONS} questions."
    for qid, q in questions.items():
        if isinstance(q, dict) and len(q.get("criteria") or ()) > MAX_OPTIONS:
            return None, f"Question '{qid}' has more than {MAX_OPTIONS} options."
    return {"state": state, "questions": questions}, None


def call(body):
    """(response dict, None), or (None, message). Never raises."""
    url, key = os.environ.get("POINTWISE_URL"), os.environ.get("POINTWISE_API_KEY")
    if not url or not key:
        return None, "The demo is not configured (missing service address or key)."
    try:
        r = requests.post(url.rstrip("/") + "/v1/systemone", json=body, timeout=TIMEOUT_S,
                          headers={"Authorization": f"Bearer {key}"})
    except requests.Timeout:
        return None, f"The service did not answer within {TIMEOUT_S} seconds. Try again."
    except requests.RequestException:
        return None, "Could not reach the service."
    if r.status_code == 200:
        try:
            return r.json(), None
        except ValueError:
            return None, "The service sent an answer the demo could not read."
    return None, ERRORS.get(r.status_code, f"The service returned an error ({r.status_code}).")


CARD = "border:1px solid var(--border-color-primary);border-radius:10px;padding:12px 14px;margin-bottom:10px;"
TRACK = "position:relative;display:block;height:6px;border-radius:999px;background:var(--border-color-primary);overflow:hidden;"


def bar(label, p, top):
    w = "600" if top else "400"
    fill = "var(--body-text-color)" if top else "var(--body-text-color-subdued)"
    return (f'<div style="display:grid;grid-template-columns:minmax(0,11rem) minmax(0,1fr) 3rem;align-items:center;gap:0 12px;'
            f'font-size:12px;line-height:22px"><span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:{w}">'
            f'{html.escape(str(label))}</span><span style="{TRACK}"><span style="position:absolute;top:0;bottom:0;left:0;'
            f'border-radius:999px;background:{fill};width:{max(0.0, min(1.0, p)) * 100:.1f}%"></span></span>'
            f'<span style="text-align:right;font-weight:{w}">{p:.2f}</span></div>')


def cards(resp, ms):
    """The service's response -> HTML, one card per question."""
    usage = resp.get("usage", {})
    out = [f'<div style="font-size:12px;margin:6px 0 10px">{len(resp["answers"])} question(s) · '
           f'{usage.get("input_tokens", "?")} input tokens · {ms:.0f} ms round trip · no text generated</div>']
    for qid, a in resp["answers"].items():
        if a["type"] == "noul":
            p = a["noul"]
            headline, detail = ("yes" if p >= 0.5 else "no"), f"p(yes) {p:.2f}"
            rows = bar("yes", p, p >= 0.5) + bar("no", 1 - p, p < 0.5)
        elif a["type"] == "choice":
            headline, detail = a["choice"], f"confidence {a['confidence']:.2f}"
            rows = "".join(bar(k, v, k == a["choice"]) for k, v in sorted(a["probabilities"].items(), key=lambda kv: -kv[1]))
        else:
            probs = a["probabilities"]
            top = max(probs, key=probs.get)
            headline, detail = f"{a['score']:.2f} of {len(probs) - 1}", f"confidence {a['confidence']:.2f}"
            rows = "".join(bar(f"{k}. {a['legend'].get(k, '')}", v, k == top) for k, v in probs.items())
        out.append(f'<div style="{CARD}"><div style="font-family:var(--font-mono);font-size:13px;font-weight:600">{html.escape(qid)}'
                   f' <span style="font-size:11px;font-weight:400">· {a["type"]}</span></div>'
                   f'<div style="font-size:17px;font-weight:600;margin:4px 0 2px">{html.escape(str(headline))}</div>'
                   f'<div style="font-size:12px;margin-bottom:8px">{detail}</div>{rows}</div>')
    return "".join(out)
