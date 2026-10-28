"""Cold-path profiler and merged-model check, on a tiny random model (CPU, no weights downloaded)."""
import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from scripts import phase6_profile as P  # noqa: E402
from scripts.merged_check import compare  # noqa: E402
from src.pointwise.model import DecisionModel  # noqa: E402
from tests.test_pointwise_serving import REQUEST, tiny_backbone, tokenizer  # noqa: E402,F401


@pytest.fixture
def lora_model(tokenizer):  # noqa: F811  function-scoped: profile() merges it in place
    backbone = peft.get_peft_model(tiny_backbone(tokenizer),
                                   peft.LoraConfig(task_type="FEATURE_EXTRACTION", r=4, target_modules=["q_proj", "v_proj"]))
    return DecisionModel(backbone.eval(), tokenizer).eval()


def test_profile_splits_each_path_into_its_stages(lora_model):
    result = P.profile(lora_model, REQUEST["state"], reps=3, warmup=1)
    assert set(result["configs"]) == {"cold", "miss", "cached", "cold_merged", "miss_merged", "cached_merged"}
    for suffix in ("", "_merged"):
        cold, miss, cached = (result["configs"][f"{n}{suffix}"]["stages"] for n in ("cold", "miss", "cached"))
        assert {"encode", "packed", "packed/mask", "packed/layer", "head", "answers"} <= set(cold)
        assert "prefill" not in cold and "questions_on_kv" not in cold
        assert {"prefill", "prefill/layer", "questions_on_kv", "questions_on_kv/mask", "questions_on_kv/layer"} <= set(miss)
        assert "prefill" not in cached and "questions_on_kv" in cached  # a hit never re-reads the state
        assert miss["prefill/layer"]["count"] == 2  # two decoder layers, one prefill pass
        assert result[f"miss_accounting{suffix}"]["prefill_ms"] > 0
    assert not isinstance(lora_model.backbone, peft.PeftModel)  # merged for the second half
    assert "| " in P.table(result)


def test_instrumentation_does_not_change_answers(lora_model):
    expected = lora_model.decide(REQUEST)
    box = [P.Recorder(lora_model.backbone.device)]
    with P.instrumented(lora_model, box):
        assert lora_model.decide(REQUEST) == expected
    assert lora_model.decide(REQUEST) == expected  # hooks and patches removed


def rows(logits_per_q):
    sources = ["banking77", "emotion", "emotion", "paws"]
    return [{"source": sources[i % 4], "id": f"r{i}", "question": 0, "logits": z, "target": [1.0, 0.0]}
            for i, z in enumerate(logits_per_q)]


def test_merged_check_ships_only_when_answers_agree():
    base = rows([[2.0, 0.0], [1.0, 0.5], [0.0, 1.0], [3.0, -1.0]] * 50)
    assert compare(base, base)["ship_merged"]
    flipped = [dict(r, logits=r["logits"][::-1]) if i % 50 == 0 else r for i, r in enumerate(base)]  # 2% flipped
    result = compare(base, flipped)
    assert result["top_agreement"] == pytest.approx(0.98) and not result["ship_merged"]
    with pytest.raises(ValueError):
        compare(base, base[:-1])
