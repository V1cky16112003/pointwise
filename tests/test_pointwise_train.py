"""Pointwise Phase 4 training, end to end on CPU with a tiny random Qwen3.

Covers the batched forward, batching and resume bookkeeping, a full train()
run that checkpoints and evaluates, resuming from that checkpoint, loss going
down on a small set, and a two-process DDP run (gloo). Skipped where torch,
peft or the cached Qwen tokenizer are missing (CI).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from src.pointwise import train as T  # noqa: E402
from src.pointwise.data import policies as P  # noqa: E402
from src.pointwise.data import sources as S  # noqa: E402
from src.pointwise.encoding import encode_request  # noqa: E402
from src.pointwise.model import DecisionModel  # noqa: E402
from src.pointwise.schema import SystemOneRequest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached locally")


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory, tokenizer):
    path = tmp_path_factory.mktemp("tiny-qwen3")
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    transformers.Qwen3Model(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("data")
    banking = S.banking77([{"text": "where is my card", "label": 0}, {"text": "top up failed", "label": 1}],
                          ["card_arrival", "top_up_failed"], __import__("random").Random(0), "heldout")
    emotion = S.emotion([{"text": "so happy", "label": 1}, {"text": "so sad", "label": 0}], ["sadness", "joy"],
                        __import__("random").Random(0), "heldout")
    files = {
        "train": P.policy_records(0, "train", 48),
        "dev": P.policy_records(0, "dev", 12),
        "heldout": banking + emotion,
    }
    for split, recs in files.items():
        with (path / f"{split}.jsonl").open("w") as f:
            for rec in recs:
                rec.pop("_text", None)
                f.write(json.dumps(rec) + "\n")
    return path


def _args(data_dir, tiny_model_dir, out, *extra):
    return ["--data-dir", str(data_dir), "--output-dir", str(out), "--model", str(tiny_model_dir),
            "--warmup", "2", "--token-budget", "1024", "--log-every", "2", "--lora-r", "4", *extra]


def test_batched_forward_matches_one_at_a_time(tiny_model_dir, tokenizer):
    backbone = transformers.AutoModel.from_pretrained(tiny_model_dir).eval()
    model = DecisionModel(backbone, tokenizer).eval()
    recs = P.policy_records(1, "dev", 4)
    encoded = [encode_request(tokenizer, SystemOneRequest.model_validate(r["request"])) for r in recs]
    with torch.no_grad():
        together = model(encoded)
        for enc, batched in zip(encoded, together):
            for a, b in zip(model([enc])[0], batched):
                torch.testing.assert_close(a, b, atol=1e-5, rtol=0)


def _fake_examples(lengths):
    class Fake:
        def __init__(self, n):
            self.length = n
    return [Fake(n) for n in lengths]


def test_make_batches_cover_everything_within_budget():
    lengths = [((i * 37) % 300) + 10 for i in range(500)]
    examples = _fake_examples(lengths)
    batches = T.make_batches(examples, token_budget=1024, seed=0)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(500))
    padded = [len(b) * max(lengths[i] for i in b) for b in batches]
    assert all(p <= 1024 or len(b) == 1 for p, b in zip(padded, batches))
    assert padded[0] == max(padded)  # biggest first: an OOM shows up at once


def test_batch_stream_crosses_epochs_and_splits_ranks():
    examples = _fake_examples([50] * 40)
    ranks = [T.BatchStream(examples, 200, seed=0, rank=r, world=2) for r in range(2)]
    per_epoch = ranks[0].steps_in(0)
    for step in range(per_epoch * 2 + 1):
        a, b = (set(map(id, s.batch(step))) for s in ranks)
        assert a and b and not a & b  # each rank gets its own batch
    assert ranks[0].steps_in(1) == ranks[0].counts[1]


def test_lr_schedule():
    assert T.lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert T.lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert T.lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert T.lr_factor(100, 10, 100) == pytest.approx(0.1)
    assert T.lr_factor(500, 10, 100) == pytest.approx(0.1)


def test_train_checkpoints_evaluates_and_resumes(tmp_path, data_dir, tiny_model_dir, capsys):
    T.main(_args(data_dir, tiny_model_dir, tmp_path, "--max-steps", "6"))
    out = tmp_path / "pointwise-train"
    state = torch.load(out / "checkpoint" / "state.pt", weights_only=False)
    assert state["step"] == 6
    result = json.loads((out / "metrics.json").read_text())
    assert result["temperature"] > 0
    assert {"banking77", "emotion", "all"} <= set(result["heldout"]["raw"])
    assert result["heldout"]["raw"]["banking77"]["chance_accuracy"] == pytest.approx(0.5)
    assert (out / "heldout_predictions.jsonl").exists()
    capsys.readouterr()

    T.main(_args(data_dir, tiny_model_dir, tmp_path, "--max-steps", "10"))
    log = capsys.readouterr().out
    assert "resumed from" in log and "at step 6" in log
    assert torch.load(out / "checkpoint" / "state.pt", weights_only=False)["step"] == 10


def test_epochs_stop_and_best_weights_are_evaluated(tmp_path, data_dir, tiny_model_dir, capsys):
    """--epochs sizes the schedule and stops there; dev is checked at every
    checkpoint (here every step) and the final evaluation uses the best weights."""
    T.main(_args(data_dir, tiny_model_dir, tmp_path, "--epochs", "1", "--checkpoint-minutes", "0",
                 "--eval-dev-records", "4"))
    out = tmp_path / "pointwise-train"
    total = int(re.search(r"LR schedule over (\d+) steps \(1.0 epochs\)", capsys.readouterr().out).group(1))
    state = torch.load(out / "checkpoint" / "state.pt", weights_only=False)
    assert state["step"] == total
    curve = [json.loads(line) for line in (out / "dev_curve.jsonl").read_text().splitlines()]
    assert [c["step"] for c in curve] == list(range(1, total + 1))
    best = min(curve, key=lambda c: c["nll"])
    result = json.loads((out / "metrics.json").read_text())
    assert result["best_step"] == best["step"] == state["best"]["step"]
    saved = torch.load(out / "best" / "state.pt", weights_only=False)
    assert saved["step"] == best["step"] and {"lora", "head"} <= set(saved)
    # The final dev NLL (T refitted on the same records) reproduces the best check's.
    assert result["dev"]["scaled"]["all"]["nll"] == pytest.approx(best["nll"], abs=1e-3)


def test_resume_search_finds_previous_session(tmp_path, data_dir, tiny_model_dir, capsys):
    previous = tmp_path / "input" / "notebook-v1"
    T.main(_args(data_dir, tiny_model_dir, previous, "--max-steps", "4"))
    capsys.readouterr()
    T.main(_args(data_dir, tiny_model_dir, tmp_path / "fresh", "--max-steps", "6", "--resume-search", str(tmp_path / "input")))
    assert "at step 4" in capsys.readouterr().out


def test_loss_goes_down_on_a_small_set(tmp_path, data_dir, tiny_model_dir, capsys):
    T.main(_args(data_dir, tiny_model_dir, tmp_path, "--max-steps", "40", "--lr", "5e-3", "--head-lr", "5e-3",
                 "--limit-train", "8", "--eval-dev-records", "2"))
    losses = [float(x) for x in re.findall(r"step \d+ loss ([\d.]+)", capsys.readouterr().out)]
    assert len(losses) == 20
    assert sum(losses[-3:]) / 3 < 0.7 * sum(losses[:3]) / 3


def test_two_process_ddp_on_cpu(tmp_path, data_dir, tiny_model_dir):
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "--master_port", "29517",
           "-m", "src.pointwise.train", *_args(data_dir, tiny_model_dir, tmp_path, "--max-steps", "4")]
    run = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    assert "world 2" in run.stdout
    assert torch.load(tmp_path / "pointwise-train" / "checkpoint" / "state.pt", weights_only=False)["step"] == 4
    assert (tmp_path / "pointwise-train" / "metrics.json").exists()


def test_augment_flag_adds_none_pairs_and_keeps_every_record(data_dir, tokenizer):
    args = T.parse_args(["--data-dir", str(data_dir), "--output-dir", "unused", "--augment"])
    plain, _ = T.load_examples(data_dir / "train.jsonl", tokenizer, args)
    aug, _ = T.load_examples(data_dir / "train.jsonl", tokenizer, args, augment=True)
    pairs = [e for e in aug if "#none-" in e.id]
    assert pairs and len(pairs) % 2 == 0
    assert {e.id for e in aug} - {e.id for e in pairs} == {e.id for e in plain}
    assert all(abs(sum(t) - 1) < 1e-6 and len(t) == len(q.option_offsets) for e in aug for t, q in zip(e.targets, e.encoded.questions))
    assert T.load_examples(data_dir / "train.jsonl", tokenizer, args, augment=True)[0][0].targets == aug[0].targets  # reproducible
