# Pointwise demo Space: design

Status: **built, with a change of hosting (2026-11-03).** Hugging Face now requires a paid plan to create a Gradio Space
(docs, "Spaces overview"), so the owner chose to run the same Gradio page as a Modal CPU function instead. The page is
`demo/pointwise_app.py` (the owner's file, used as the base), its logic is in `demo/pointwise_client.py`, and it is
deployed by `deploy/modal_demo.py`; see `docs/modal-demo.md`. Wherever this document says Space, read "the Modal
function"; there is no `space/` folder and no Space README.

## Why

A recruiter who is deciding whether to interview the owner as an AI engineer opens one link and, within a minute, sees
that Pointwise is a working, deployed system and not a notebook. Kev's public demo (a Gradio Hugging Face Space with
presets and probability bars) is the model: clean, live and professional. The owner chose a Gradio app like Kev's.

## Goals

- One public URL a recruiter can open without installing or signing up.
- Pick a preset, click run, see a card per question with the answer, its confidence and one bar per option.
- Every result comes from the real service on Modal (`docs/modal.md`), so the demo proves the deployment.
- The first visitor after the GPU has been idle sees a plain "Waking up the GPU, about 20 seconds" message and waits.
  The owner decided this is fine; the GPU is not kept warm.
- Visitors cannot read the API key.
- The page is honest about what the model is: it states the main limitation (confidence is only roughly calibrated, ECE
  0.056, from `docs/production.md`) and links the repo, the README and `docs/production.md`.

## Non-goals

- No model inside the Space. No ZeroGPU (hosting it needs a paid plan, and the owner will pay nothing).
- No accounts, no per-user keys, no history, no database.
- No Next.js playground, chess demo or other extras from Kev.
- No keep-warm job, no auto-scaling changes, no custom domain.
- No retraining or retuning (project rule).

## Anti-goals (a failure even if it technically works)

- A visitor, or a bot, runs up the Modal bill. The spend limit is the backstop; the Space must also limit abuse (below).
- The page shows made-up or illustrative numbers as if measured. Any number on the page comes from a real response or a
  named report.
- The API key leaks into the page, the browser, logs or the repo.
- The recruiter sees a blank page or a stack trace instead of an explanation when Modal is asleep, slow or out of credit.

## What the page contains (modelled on Kev's `app.py`)

1. Title, one-sentence description ("typed questions in, calibrated probabilities out; no text generated"), and links to
   the GitHub repo and the docs.
2. Three preset buttons, chosen from the sources where Pointwise scored best (`README.md`, Results), not from Kev's
   presets. The owner decided **not to test them against the live endpoint first**, so the answers are unverified:
   - Customer-message routing, in the style of Banking77 (a `choice` question; 0.626 zero-shot).
   - "Does this passage answer the question?", in the style of QNLI (a `noul` question; 0.818, its best source).
   - Short-text sentiment, in the style of TweetEval (a `choice` question; 0.652).
   Presets must use only the question types and limits the server accepts (`src/pointwise/schema.py`). The README
   support-triage example is left out, because the live test gave it a 0.32 confidence headline. No `score` preset,
   because its nearest task (Emotion, 0.492) is the weakest.
3. Two editable boxes: the state (text) and the questions (JSON). The JSON is the real request body, so a technical
   visitor can edit it.
4. A Run button and a status line: "Waking up the GPU, about 20 seconds" while waiting, then input tokens and server time.
5. Result cards: question id, type, headline answer, confidence, bars sorted by probability (as in Kev's `render_answers`).
6. A collapsed "Limitations" block, drawn from `docs/production.md` section 7, and a collapsed "Raw response" JSON.

Skipped from Kev's Space: the calibrated/raw toggle (Pointwise's server always applies T 1.516) and the option-order
stability check (would need the server to expose it). Say so if you want either.

## How it works

```
Browser -> Gradio Space (free CPU) -> HTTPS POST /v1/systemone + Bearer key -> Modal T4 -> JSON -> cards
```

- The Space is a thin client. It validates the visitor's input, calls the Modal URL with `requests`/`httpx`, and renders
  the response. The URL and key are Space secrets (`POINTWISE_URL`, `POINTWISE_API_KEY`), read from the environment.
- The Space is a Docker-free Gradio Space: `app.py`, `presets.py`, `requirements.txt` and a README with the Space YAML
  header. It lives in `space/` in this repo (as Kev does) and is pushed to the Space's own git repo by hand. The owner
  does the publishing so they learn it.
- The wait: the first call after idle takes about 20 s (measured, `docs/modal.md`), so the request timeout is set to
  about 90 s, and the status line says the GPU is waking. Errors become short messages: timeout, 401 (misconfigured key),
  503 `overloaded` (try again in a few seconds), and anything else.
- Abuse limits live in the Space, since the server has no per-user limits: cap state length and question count to the
  server's own limits or lower, cap output size, keep Gradio's queue small (concurrency 1-2), and rate-limit per visitor
  if Gradio's request object makes that cheap. The Modal spend limit stays the hard stop.

## Constraints

- Free Hugging Face CPU Space, public. No card.
- `/healthz` on Modal needs no key and wakes the GPU; the Space must never ping it on a timer.
- The repo's commit and push rules apply: `/code-review` before commits unless the owner skips it, no `git add .`,
  push only on the owner's word. `SESSION_HANDOFF.md` and `docs/designs/pointwise-next-steps.md` are never committed.
- Gradio is pinned to an exact version, and `requirements.txt` for the Space is separate from `requirements-serve.txt`.

## Decided by the owner

- A Gradio Space, like Kev's.
- Link the GitHub repo (`github.com/V1cky16112003/pointwise`). No name, contact number or contact link on the page.
- No testing of the presets against the live endpoint before building.
- The 20 s wake-up wait is acceptable; show a message.
- The owner publishes by hand, guided.
- Plan first, nothing built until approval.

## My calls (the owner can change any of these)

- The Space calls Modal; it does not host the model.
- `space/` folder in this repo, not a separate repo.
- Three presets from the strong sources (above), not Kev's four; no stability toggle or calibration toggle.
- Because the presets are not tested before release, each preset's blurb says what the task is and does not promise a
  particular answer. The page shows whatever the model returns.
- Request timeout about 90 s; Gradio concurrency 2.
- Show latency as "server time", not "time on GPU", because the client cannot tell GPU time from queueing.

## Left to the builder

- Exact Gradio layout and styling, within "clean, close to Kev".
- Preset wording, as long as each preset is a valid request and the answers look sensible when run.
- The per-visitor rate limit, if it takes more than a few lines.

## How we know it works

- A unit test that every preset parses as a valid `SystemOneRequest` (import from `src/pointwise/schema.py`).
- A test of the response-to-cards function on a recorded real response (saved from the Modal run), and on the error cases.
- Run the Space locally against the deployed Modal URL, then after publishing, open the public URL from a private window
  and run each preset, including once after the GPU has gone idle. This is the owner's first look at the presets'
  answers; they are not run earlier.
- Check the page source and network tab for the key.

## Open questions for the owner

None. Risk the owner accepted: a preset may return a wrong or low-confidence answer in front of a recruiter, and the
first time anyone sees it will be the owner's own first run after publishing. If one looks bad then, swap the preset
text in `space/presets.py` and push again.
