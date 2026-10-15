"""No-training and task-specific baselines for Phase 5 (docs/designs/pointwise-plan.md).

A causal LM reads the same request as Pointwise, written as plain text, and every option is
scored by its full-sequence log-likelihood (option tokens + EOS), as in src/readout.py. The
scores go in a row's `logits`, so softmax renormalises over the candidates and the arm is
scored by the same code as Pointwise.
"""
from __future__ import annotations

import torch

from src.pointwise.schema import SystemOneRequest, question_options, render
from src.readout import _token_logprobs, repeat_cache

OPTION_CHUNK = 32  # options scored per forward pass; each row carries a copy of the prompt cache


def question_prompt(request: SystemOneRequest, key: str) -> tuple[str, list[str]]:
    """(prompt, continuations) for one question. Options are listed with their texts and answered by
    name; a noul question is answered "no" or "yes"."""
    question = request.questions[key]
    if question.type == "score":
        raise NotImplementedError("score questions need numbered options in the prompt; Phase 5 only has choice and noul")
    names, texts = question_options(question)
    answers = ["no", "yes"] if question.type == "noul" else names
    prompt = "\n\n".join([
        render(request.state),
        f"Question: {render(question.instructions)}",
        "Options:\n" + "\n".join(f"- {t}" for t in texts) + "\nAnswer:",
    ])
    return prompt, [" " + a for a in answers]


@torch.no_grad()
def score_options(model, tokenizer, prompt: str, continuations: list[str]) -> list[float]:
    """Sequence log-likelihood of each continuation after the prompt. The prompt is read once."""
    device = model.device
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    out = model(input_ids=torch.tensor([prompt_ids], device=device), use_cache=True)
    first = out.logits[0, -1].float().log_softmax(-1)
    seqs = [tokenizer.encode(c, add_special_tokens=False) + [tokenizer.eos_token_id] for c in continuations]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    scores: list[float] = []
    for start in range(0, len(seqs), OPTION_CHUNK):
        chunk = seqs[start : start + OPTION_CHUNK]
        width = max(map(len, chunk))
        ids = torch.tensor([s + [pad] * (width - len(s)) for s in chunk], device=device)
        mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in chunk], device=device)
        # Right padding is safe without an attention mask: causal attention never lets a pad affect an earlier token.
        logits = model(input_ids=ids[:, :-1], past_key_values=repeat_cache(out.past_key_values, len(chunk))).logits
        rest = (_token_logprobs(logits, ids[:, 1:]) * mask[:, 1:]).sum(1)
        scores += (first[ids[:, 0]] + rest).tolist()
    return scores


def score_record(model, tokenizer, rec: dict) -> list[dict]:
    """One row per choice/noul question of a Phase 3 record, in the held-out predictions' shape. Score questions
    (policies, SST-5, Yelp; dev only) are skipped: the report compares arms on the questions they all answer."""
    request = SystemOneRequest.model_validate(rec["request"])
    rows = []
    for k, key in enumerate(request.questions):
        if request.questions[key].type == "score":
            continue
        prompt, continuations = question_prompt(request, key)
        rows.append({"source": rec["source"], "id": rec["id"], "question": k,
                     "logits": score_options(model, tokenizer, prompt, continuations),
                     "target": rec["targets"][key]})
    return rows
