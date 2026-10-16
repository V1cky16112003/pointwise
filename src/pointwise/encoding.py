"""Token layout of a decision request (Pointwise Phase 2).

    state:      <state> state tokens
    question k: <q> instruction <opt> option 1 </opt> ... <opt> option K </opt> <decide>

The read-out (src/pointwise/model.py) compares the hidden state at <decide> with
the hidden state at each option's </opt>, the last token of that option, which
has read the whole option text.

Delimiters reuse Qwen special tokens that a decision model never otherwise
sees, so no embedding rows are added (Kev does the same); LoRA learns what they
mean. Caller text is tokenised with special tokens split apart, so a state or
option that *contains* "<|box_end|>" can never close an option early or add
one: option boundaries come only from this file.

Tokenizer-only, no torch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.pointwise.schema import SystemOneRequest, question_options, render

STATE, QUESTION, OPTION_OPEN, OPTION_CLOSE, DECIDE = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)
DELIMITERS = (STATE, QUESTION, OPTION_OPEN, OPTION_CLOSE, DECIDE)
# Jev's documented limit: a 32k-token state, plus the longest single question.
MAX_STATE_TOKENS = 32768

_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class UnsafeText(ValueError):
    """Caller text that cannot be tokenised without producing a delimiter token. A server turns it into a 422."""


class ContextOverflow(ValueError):
    """The state is longer than the model accepts. A server turns it into a 422."""


@dataclass(frozen=True)
class EncodedQuestion:
    ids: list[int]  # <q> ... <decide>
    decide_offset: int  # index of <decide> within ids
    option_offsets: list[int]  # index of each option's </opt> within ids


@dataclass(frozen=True)
class EncodedRequest:
    state_ids: list[int]
    questions: list[EncodedQuestion]  # in request order


def delimiter_ids(tokenizer) -> list[int]:
    ids = [tokenizer.convert_tokens_to_ids(t) for t in DELIMITERS]
    if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in ids:
        raise ValueError("tokenizer lacks the Qwen delimiter tokens")
    return ids


def user_tokens(tokenizer, text: str) -> list[int]:
    """Tokenise caller text so it can never produce a delimiter token. Splitting
    special tokens keeps the text exact; if a tokenizer ignores that flag, fall
    back to rewriting "<|name|>" as "<¦name¦>". Raises rather than ever letting
    a delimiter through."""
    forbidden = set(delimiter_ids(tokenizer))
    ids = tokenizer(text, add_special_tokens=False, split_special_tokens=True).input_ids
    if forbidden.isdisjoint(ids):
        return ids
    ids = tokenizer(_SPECIAL.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids
    if not forbidden.isdisjoint(ids):
        raise UnsafeText("could not tokenise caller text without delimiter tokens")
    return ids


def encode_request(
    tokenizer, request: SystemOneRequest, max_state_tokens: int = MAX_STATE_TOKENS
) -> EncodedRequest:
    state_tok, question_tok, open_tok, close_tok, decide_tok = delimiter_ids(tokenizer)
    state_ids = [state_tok] + user_tokens(tokenizer, render(request.state))
    if len(state_ids) > max_state_tokens:
        raise ContextOverflow(
            f"state is {len(state_ids):,} tokens, over the {max_state_tokens:,}-token limit"
        )
    questions = []
    for question in request.questions.values():
        _, texts = question_options(question)
        ids = [question_tok] + user_tokens(tokenizer, render(question.instructions))
        option_offsets = []
        for text in texts:
            ids += [open_tok] + user_tokens(tokenizer, text) + [close_tok]
            option_offsets.append(len(ids) - 1)
        ids.append(decide_tok)
        questions.append(EncodedQuestion(ids, len(ids) - 1, option_offsets))
    return EncodedRequest(state_ids, questions)
