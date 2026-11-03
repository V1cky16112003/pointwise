"""Gradio demo for the deployed Pointwise service (docs/modal.md, docs/modal-demo.md). It calls POST /v1/systemone over
HTTP and holds the API key server-side, so visitors never see it.

    POINTWISE_URL=https://<you>--pointwise-serve.modal.run POINTWISE_API_KEY=... python demo/pointwise_app.py

Cost guard: each visitor IP gets RATE_LIMIT requests per minute, since every request can wake the paid GPU. The limit is
in memory and per process (shortcut: resets on restart and ignores proxies' real IPs; upgrade if the site gets traffic).
Input caps and the 90 s timeout are in pointwise_client.py.
"""
import json
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import gradio as gr

from pointwise_client import build_body, call, cards

REPO = "https://github.com/V1cky16112003/pointwise"
RATE_LIMIT = int(os.environ.get("DEMO_RATE_LIMIT", "6"))  # requests per IP per minute
_seen: dict[str, deque] = defaultdict(deque)
POOL = ThreadPoolExecutor(max_workers=2)

# Chosen from the sources Pointwise scored best on (README, Results). Not run against the live service before release.
PRESETS = {
    "Route a customer message": {
        "state": "My new card still has not arrived, and it has been two weeks since I ordered it.",
        "questions": {
            "intent": {"type": "choice", "instructions": "What is the customer asking about?",
                       "criteria": {"card_not_arrived": "A card that was ordered has not been delivered",
                                    "lost_or_stolen_card": "The card is lost or stolen",
                                    "duplicate_charge": "The customer was charged twice",
                                    "change_pin": "The customer wants to change a PIN",
                                    "top_up": "The customer wants to add money to the account"}},
        },
    },
    "Does the passage answer the question?": {
        "state": {"question": "What is the capital of Australia?",
                  "passage": "Canberra, the capital of Australia, was chosen in 1908 as a compromise between Sydney and Melbourne."},
        "questions": {"answers_the_question": {"type": "noul", "instructions": "Does the passage contain the answer to the question?"}},
    },
    "Sentiment of a short text": {
        "state": "Finally got my order and it is even better than the photos. Really happy with it!",
        "questions": {
            "sentiment": {"type": "choice", "instructions": "What is the sentiment of this message?",
                          "criteria": {"negative": None, "neutral": None, "positive": None}},
            "complaint": {"type": "noul", "instructions": "Is the writer complaining?"},
        },
    },
}
FIRST = next(iter(PRESETS))


def too_fast(ip: str, now: float) -> bool:
    q = _seen[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


def as_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, indent=2)


def run(state: str, questions_json: str, request: gr.Request):
    """Yields (status, cards html, raw response)."""
    if too_fast(request.client.host if request else "?", time.time()):
        yield f"**Rate limit: {RATE_LIMIT} requests per minute. Try again shortly.**", "", ""
        return
    body, err = build_body(state, questions_json)
    if err:
        yield f"**{err}**", "", ""
        return
    t0 = time.perf_counter()
    fut = POOL.submit(call, body)
    try:
        resp, err = fut.result(timeout=3)
    except TimeoutError:  # no answer after 3 s: the GPU was asleep
        yield "Waking up the GPU, about 20 seconds...", "", ""
        resp, err = fut.result()
    if err:
        yield f"**{err}**", "", ""
        return
    yield "Done.", cards(resp, (time.perf_counter() - t0) * 1000), json.dumps(resp, indent=2)


def build() -> gr.Blocks:
    with gr.Blocks(title="Pointwise") as demo:
        gr.Markdown("# Pointwise\nTyped questions in, calibrated probabilities out. One forward pass, no text generated. "
                    "A fine-tuned Qwen3-1.7B on a Modal T4. "
                    f"[Code and docs]({REPO}) · [Production notes]({REPO}/blob/main/docs/production.md)\n\n"
                    "_The first request after a quiet minute can take about 20 seconds while the GPU wakes._")
        preset = gr.Radio(list(PRESETS), value=FIRST, label="Example")
        state = gr.Textbox(as_text(PRESETS[FIRST]["state"]), label="State (text, or a JSON object)", lines=5)
        questions = gr.Code(json.dumps(PRESETS[FIRST]["questions"], indent=2), language="json", label="Questions (JSON)")
        go = gr.Button("Run", variant="primary")
        status, out = gr.Markdown(), gr.HTML()
        with gr.Accordion("Raw response", open=False):
            raw = gr.Code(language="json")
        with gr.Accordion("Limitations", open=False):
            gr.Markdown("- Confidence is only roughly calibrated: expected calibration error 0.056 out of domain, just over the "
                        "0.05 target. Do not use it to pick out a safe subset.\n"
                        "- Weak on emotion and paraphrase tasks. Accuracy on texts longer than 384 tokens was never measured.\n"
                        f"- One training run, one seed. Full table: [docs/production.md]({REPO}/blob/main/docs/production.md).")
        preset.change(lambda p: (as_text(PRESETS[p]["state"]), json.dumps(PRESETS[p]["questions"], indent=2)), preset, [state, questions])
        go.click(run, [state, questions], [status, out, raw])
    return demo


if __name__ == "__main__":
    if not os.environ.get("POINTWISE_URL") or not os.environ.get("POINTWISE_API_KEY"):
        raise SystemExit("Set POINTWISE_URL and POINTWISE_API_KEY")
    build().queue(default_concurrency_limit=2).launch()
