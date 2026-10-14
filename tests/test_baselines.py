"""Cached option scoring must equal scoring each full prompt+option sequence on its own."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.pointwise import baselines as B
from src.pointwise.schema import SystemOneRequest


class FakeTokenizer:
    """Bytes as token ids (+1); id 0 is both EOS and pad."""

    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [b + 1 for b in text.encode()]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = transformers.Qwen3Config(vocab_size=260, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=2, num_key_value_heads=1, head_dim=16)
    return transformers.Qwen3ForCausalLM(config).eval()


def naive(model, tok, prompt, continuation):
    p = tok.encode(prompt)
    c = tok.encode(continuation) + [tok.eos_token_id]
    ids = torch.tensor([p + c])
    logp = model(input_ids=ids).logits[0].log_softmax(-1)
    return sum(logp[len(p) + i - 1, t].item() for i, t in enumerate(c))


def test_score_options_matches_naive_for_options_of_different_lengths(model):
    tok = FakeTokenizer()
    options = [" a", " bb", " a longer option", " c"]
    got = B.score_options(model, tok, "State text\n\nAnswer:", options)
    want = [naive(model, tok, "State text\n\nAnswer:", o) for o in options]
    assert got == pytest.approx(want, abs=1e-4)


def test_score_options_chunks_agree(model, monkeypatch):
    tok = FakeTokenizer()
    options = [f" option {i}" for i in range(7)]
    whole = B.score_options(model, tok, "Pick one.\nAnswer:", options)
    monkeypatch.setattr(B, "OPTION_CHUNK", 3)
    assert B.score_options(model, tok, "Pick one.\nAnswer:", options) == pytest.approx(whole, abs=1e-4)


def test_question_prompt_for_noul_and_choice():
    noul = SystemOneRequest.model_validate({"state": {"a": "x"}, "questions": {"q": {"type": "noul", "instructions": "Same?"}}})
    prompt, cont = B.question_prompt(noul, "q")
    assert cont == [" no", " yes"] and prompt.endswith("- no\n- yes\nAnswer:") and "a: x" in prompt
    choice = SystemOneRequest.model_validate(
        {"state": "t", "questions": {"q": {"type": "choice", "instructions": "Pick", "criteria": {"red": "a colour", "blue": None}}}}
    )
    prompt, cont = B.question_prompt(choice, "q")
    assert cont == [" red", " blue"] and "- red: a colour" in prompt
