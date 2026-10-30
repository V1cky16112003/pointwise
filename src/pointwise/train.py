"""Pointwise Phase 4: RLCD-lite training (docs/designs/pointwise-phase4-plan.md).

LoRA (r=16, attention + MLP) on a frozen Qwen3 base, plus the fp32 pointer
head, trained with the log score (soft cross-entropy) against each question's
target distribution, optionally plus a Brier term. Both are strictly proper:
the expected loss is lowest when the predicted distribution equals the true one.

Built for Kaggle's limits:
- Multi-GPU through torchrun (DDP), one process per T4.
- Stops itself after --max-hours and saves, because Kaggle keeps no output from
  a session that times out. Also checkpoints every --checkpoint-minutes.
- Resumes from the newest checkpoint found under --resume-search (attach the
  previous notebook version as input), keeping the step, optimiser and LR
  schedule.
- The LR schedule's length is set from the measured step time and
  --planned-hours (the training time across all sessions), so cosine decay ends
  when the planned sessions do; or from --epochs, which also stops the run there.
- The largest batch runs first, so an out-of-memory error shows up in the first
  minute, not hours in.

At every checkpoint, rank 0 also scores dev (NLL after fitting T) and keeps the
best weights so far (dev_curve.jsonl logs each check). At the end it scores dev
and held-out sets with the best weights, fits a temperature on dev, and writes
metrics.json, the held-out predictions and best/state.pt (adapter + head).

Usage:
    torchrun --nproc_per_node 2 -m src.pointwise.train --data-dir data/pointwise --output-dir out
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from src.pointwise import metrics as M
from src.pointwise.data import records as R
from src.pointwise.encoding import EncodedRequest, encode_request
from src.pointwise.model import DecisionModel
from src.pointwise.schema import SystemOneRequest

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CHUNK = 1000  # records sorted by length within a chunk, so batches carry little padding
SPEED_STEPS = (10, 60)  # steps timed to size the LR schedule (inside the warmup)


@dataclass
class Example:
    encoded: EncodedRequest
    targets: list[list[float]]  # one distribution per question, request order
    source: str
    id: str

    @property
    def length(self) -> int:
        return len(self.encoded.state_ids) + sum(len(q.ids) for q in self.encoded.questions)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--augment", action="store_true",
                   help="Kev's option augmentation on train records: none option, distractors, none pairs")
    p.add_argument("--p-none-pair", type=float, default=0.25, help="with --augment: chance a record adds a none pair")
    p.add_argument("--brier-weight", type=float, default=0.0)
    p.add_argument("--token-budget", type=int, default=4096, help="padded tokens per batch per GPU")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--max-hours", type=float, default=10.5, help="training time in this session")
    p.add_argument("--planned-hours", type=float, default=21.0, help="training time across all sessions")
    p.add_argument("--max-steps", type=int, default=None, help="stop after this global step (tests, smoke runs)")
    p.add_argument("--epochs", type=float, default=None,
                   help="train this many epochs; sizes the LR schedule instead of --planned-hours")
    p.add_argument("--checkpoint-minutes", type=float, default=30.0, help="also how often dev loss is checked")
    p.add_argument("--resume-search", type=Path, default=None)
    p.add_argument("--max-state", type=int, default=384)
    p.add_argument("--max-branch", type=int, default=1024, help="state + one question")
    p.add_argument("--max-packed", type=int, default=2048, help="state + all questions")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--eval-dev-records", type=int, default=1500)
    p.add_argument("--eval-heldout-per-source", type=int, default=500)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# --- data ------------------------------------------------------------------------


def augmented(rec: dict, seed: int, p_none_pair: float) -> list[dict]:
    """The record after Kev's augmentation, plus its none pair with probability p_none_pair.
    shortcut: drawn once per run, not per epoch as Kev does; redraw per epoch if 2 epochs overfit the augmentation."""
    rng = random.Random(f"{seed}:{rec['id']}")
    out = [R.augment(rec, rng)]
    if rng.random() < p_none_pair:
        out += R.none_pair(rec, rng)
    return out


def load_examples(path: Path, tokenizer, args, limit: int | None = None, augment: bool = False) -> tuple[list[Example], dict]:
    """Encode records and drop any that don't fit the training context. Records
    are dropped whole rather than truncated: cutting a state could cut the
    evidence a label depends on."""
    examples, dropped = [], {}
    with path.open(encoding="utf-8") as f:
        lines = [line for n, line in enumerate(f) if limit is None or n < limit]
    for line in lines:
        recs = [json.loads(line)]
        if augment:
            recs = augmented(recs[0], args.seed, args.p_none_pair)
        for rec in recs:
            request = SystemOneRequest.model_validate(rec["request"])
            encoded = encode_request(tokenizer, request)
            state = len(encoded.state_ids)
            branches = [len(q.ids) for q in encoded.questions]
            if state > args.max_state or state + max(branches) > args.max_branch or state + sum(branches) > args.max_packed:
                dropped[rec["source"]] = dropped.get(rec["source"], 0) + 1
                continue
            targets = [rec["targets"][key] for key in request.questions]
            examples.append(Example(encoded, targets, rec["source"], rec["id"]))
    return examples, dropped


def make_batches(examples: list[Example], token_budget: int, seed: int) -> list[list[int]]:
    """Index batches for one epoch: shuffle, sort by length within chunks, fill
    batches up to token_budget padded tokens (batch size x longest example),
    then shuffle the batch order. The largest batch goes first, to surface an
    out-of-memory error immediately."""
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    batches: list[list[int]] = []
    for start in range(0, len(order), CHUNK):
        chunk = sorted(order[start : start + CHUNK], key=lambda i: examples[i].length)
        batch: list[int] = []
        for i in chunk:
            longest = examples[i].length  # ascending within the chunk
            if batch and longest * (len(batch) + 1) > token_budget:
                batches.append(batch)
                batch = []
            batch.append(i)
        if batch:
            batches.append(batch)
    rng.shuffle(batches)
    biggest = max(range(len(batches)), key=lambda b: len(batches[b]) * max(examples[i].length for i in batches[b]))
    batches.insert(0, batches.pop(biggest))
    return batches


class BatchStream:
    """Global batch index -> this rank's batch. Epoch e reshuffles with seed + e,
    so any step can be recomputed on resume without saving the data order."""

    def __init__(self, examples: list[Example], token_budget: int, seed: int, rank: int, world: int):
        self.examples, self.budget, self.seed, self.rank, self.world = examples, token_budget, seed, rank, world
        self.epochs: dict[int, list[list[int]]] = {}  # only the current epoch's batches
        self.counts: dict[int, int] = {}  # steps per epoch, for every epoch seen

    def _epoch(self, epoch: int) -> list[list[int]]:
        if epoch not in self.epochs:
            batches = make_batches(self.examples, self.budget, self.seed + epoch)
            self.epochs = {epoch: batches[: len(batches) - len(batches) % self.world]}  # one per rank per step
        return self.epochs[epoch]

    def steps_in(self, epoch: int) -> int:
        """Steps in an epoch. Epochs differ slightly (different shuffles), so each
        is counted from its own batches, once."""
        if epoch not in self.counts:
            self.counts[epoch] = len(self._epoch(epoch)) // self.world
        return self.counts[epoch]

    def batch(self, step: int) -> list[Example]:
        epoch = 0
        while step >= (n := self.steps_in(epoch)):
            step -= n
            epoch += 1
        indices = self._epoch(epoch)[step * self.world + self.rank]
        return [self.examples[i] for i in indices]


# --- model ------------------------------------------------------------------------


def build_model(args, device: torch.device) -> DecisionModel:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    # T4: no bf16. MPS (Apple GPU) serves in fp16 too: fp32 would not fit an 8 GB Mac. Training never runs on MPS.
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    backbone = AutoModel.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to(device)
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.config.use_cache = False
    config = LoraConfig(
        task_type="FEATURE_EXTRACTION",
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS,
    )
    torch.manual_seed(args.seed)  # same LoRA and head init on every rank
    return DecisionModel(get_peft_model(backbone, config), tokenizer)


def question_loss(logits: torch.Tensor, target: torch.Tensor, brier_weight: float) -> torch.Tensor:
    """Log score -sum q log p, plus an optional Brier term. Both strictly proper."""
    log_p = torch.log_softmax(logits, -1)
    loss = -(target * log_p).sum()
    if brier_weight:
        loss = loss + brier_weight * (log_p.exp() - target).square().sum()
    return loss


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup, then cosine from 1 down to 0.1 at `total`, flat after."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


# --- checkpoints ---------------------------------------------------------------------


def trainable_state(model: DecisionModel) -> dict:
    from peft import get_peft_model_state_dict

    return {"lora": get_peft_model_state_dict(model.backbone), "head": model.head.state_dict()}


def save_checkpoint(path: Path, model, optimizer, scaler, step: int, total_steps: int, args, extra: dict | None = None) -> None:
    """Written to a temporary file then renamed, so a kill mid-save never leaves
    a half-written checkpoint where the resume search would find it. `extra`
    carries the best weights so far and the dev curve, so a resume keeps them."""
    path.mkdir(parents=True, exist_ok=True)
    state = {
        **trainable_state(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "total_steps": total_steps,
        "args": {k: str(v) for k, v in vars(args).items()},
        **(extra or {}),
    }
    _save(state, path)


def _save(state: dict, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / "state.pt.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path / "state.pt")


def find_checkpoint(*roots: Path | None) -> Path | None:
    """The checkpoint with the highest step under any of `roots`."""
    best, best_step = None, -1
    for root in roots:
        if root is None or not root.exists():
            continue
        # minijev-train: Phase 4 session 1 ran before the rename to Pointwise.
        paths = [*root.glob("**/pointwise-train/checkpoint/state.pt"),
                 *root.glob("**/minijev-train/checkpoint/state.pt")]
        for path in paths:
            step = torch.load(path, map_location="cpu", weights_only=False)["step"]
            if step > best_step:
                best, best_step = path, step
    return best


def load_trainable(model: DecisionModel, state: dict) -> None:
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model.backbone, state["lora"])
    model.head.load_state_dict(state["head"])


def load_checkpoint(path: Path, model, optimizer, scaler) -> dict:
    """Restores model, optimiser and scaler; returns the whole state for step,
    schedule length, best weights and dev curve."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    load_trainable(model, state)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    return state


# --- evaluation ------------------------------------------------------------------------


@torch.no_grad()
def collect_logits(model: DecisionModel, examples: list[Example], token_budget: int) -> list[dict]:
    """One row per question: source, id, key index, logits, target."""
    if not examples:
        return []
    model.eval()
    rows = []
    for indices in make_batches(examples, token_budget, seed=0):
        batch = [examples[i] for i in indices]
        with torch.autocast(model.backbone.device.type, dtype=torch.float16, enabled=model.backbone.device.type == "cuda"):
            logits = model([ex.encoded for ex in batch])
        for ex, per_question in zip(batch, logits):
            for k, (z, target) in enumerate(zip(per_question, ex.targets)):
                rows.append({"source": ex.source, "id": ex.id, "question": k, "logits": z.tolist(), "target": target})
    model.train()
    return rows


def score_rows(rows: list[dict], temperature: float) -> dict:
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(M.question_scores(r["logits"], r["target"], temperature))
    out = {source: M.summarise(scores) for source, scores in sorted(by_source.items())}
    out["all"] = M.summarise([s for scores in by_source.values() for s in scores])
    chance = {}
    for r in rows:
        chance.setdefault(r["source"], []).append(1 / len(r["target"]))
    for source, values in chance.items():
        out[source]["chance_accuracy"] = sum(values) / len(values)
    return out


def evaluate(model, dev: list[Example], heldout: list[Example], args, out_dir: Path) -> dict:
    dev_rows = collect_logits(model, dev, args.token_budget)
    heldout_rows = collect_logits(model, heldout, args.token_budget)
    temperature = M.fit_temperature([r["logits"] for r in dev_rows], [r["target"] for r in dev_rows])
    ood = [r for r in heldout_rows if r["source"] != "banking77"]
    result = {
        "temperature": temperature,
        "dev": {"raw": score_rows(dev_rows, 1.0), "scaled": score_rows(dev_rows, temperature)},
        "heldout": {"raw": score_rows(heldout_rows, 1.0), "scaled": score_rows(heldout_rows, temperature)},
    }
    scaled_ood = M.summarise([M.question_scores(r["logits"], r["target"], temperature) for r in ood]) if ood else {"n": 0}
    result["gate"] = {
        "heldout_ood_ece_scaled": scaled_ood.get("ece"),
        "pass_ece": scaled_ood.get("ece") is not None and scaled_ood["ece"] <= 0.05,
    }
    with (out_dir / "heldout_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in heldout_rows:
            f.write(json.dumps(r) + "\n")
    return result


def dev_check(model, dev: list[Example], args) -> dict:
    """Dev NLL after fitting T, which is what the best checkpoint is chosen on:
    T is fitted at the end anyway, so raw overconfidence alone shouldn't pick
    an early checkpoint over a better-ranking later one."""
    rows = collect_logits(model, dev, args.token_budget)
    logits, targets = [r["logits"] for r in rows], [r["target"] for r in rows]
    temperature = M.fit_temperature(logits, targets)
    raw = M.summarise([M.question_scores(z, q) for z, q in zip(logits, targets)])
    scaled = M.summarise([M.question_scores(z, q, temperature) for z, q in zip(logits, targets)])
    return {"nll": scaled["nll"], "raw_nll": raw["nll"], "accuracy": raw["accuracy"],
            "temperature": temperature, "ece": scaled["ece"], "raw_ece": raw["ece"]}


def _stratified(examples: list[Example], per_source: int) -> list[Example]:
    taken: dict[str, int] = {}
    out = []
    for ex in examples:
        if taken.get(ex.source, 0) < per_source:
            taken[ex.source] = taken.get(ex.source, 0) + 1
            out.append(ex)
    return out


# --- main ----------------------------------------------------------------------------------


def main(argv=None) -> None:
    args = parse_args(argv)
    started = time.time()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        # Long timeout: the other ranks wait at the final barrier while rank 0 evaluates.
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=datetime.timedelta(hours=2))
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    say = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    out_dir = args.output_dir / "pointwise-train"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args, device)
    tokenizer = model.tokenizer
    train, dropped = load_examples(args.data_dir / "train.jsonl", tokenizer, args, args.limit_train, args.augment)
    say(f"train examples {len(train)}, dropped for context {dropped}, world {world}, setup {time.time() - started:.0f}s")

    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr}, {"params": model.head.parameters(), "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    step, total_steps = 0, None
    best, curve = None, []  # best: trainable weights (on CPU) with the lowest dev NLL so far
    resume = find_checkpoint(out_dir.parent, args.resume_search)
    if resume is not None:
        state = load_checkpoint(resume, model, optimizer, scaler)
        step, total_steps, best, curve = state["step"], state["total_steps"], state.get("best"), state.get("curve", [])
        say(f"resumed from {resume} at step {step}, schedule length {total_steps}")

    ddp = (
        torch.nn.parallel.DistributedDataParallel(model, device_ids=None, find_unused_parameters=False)
        if distributed
        else model
    )
    stream = BatchStream(train, args.token_budget, args.seed, rank, world)
    say(f"steps in epoch 0: {stream.steps_in(0)}")
    if args.epochs is not None and total_steps is None:
        total_steps = max(1, round(args.epochs * stream.steps_in(0)))
        say(f"LR schedule over {total_steps} steps ({args.epochs} epochs)")
    dev, eval_rng = [], random.Random(args.seed)  # one rng for dev then held-out: same samples as session 1
    if rank == 0:
        dev, _ = load_examples(args.data_dir / "dev.jsonl", tokenizer, args)
        eval_rng.shuffle(dev)
        dev = dev[: args.eval_dev_records]

    def check_dev() -> None:
        """Rank 0: score dev, log it, keep a CPU copy of the weights if best so far."""
        nonlocal best
        result = {"step": step, **dev_check(model, dev, args)}
        curve.append(result)
        with (out_dir / "dev_curve.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        if best is None or result["nll"] < best["nll"]:
            weights = {k: {n: t.detach().cpu().clone() for n, t in v.items()} for k, v in trainable_state(model).items()}
            best = {"step": step, "nll": result["nll"], **weights}
        say(f"dev at step {step}: nll {result['nll']:.4f} (raw {result['raw_nll']:.4f}, T {result['temperature']:.2f}) "
            f"acc {result['accuracy']:.4f}; best step {best['step']}")

    model.train()

    train_start = time.time()
    last_checkpoint = time.time()
    window_loss, window_questions, window_start, speed_start = 0.0, 0, time.time(), None
    while True:
        out_of_time = (time.time() - train_start) / 3600 > args.max_hours
        at_limit = (args.max_steps is not None and step >= args.max_steps) or (
            args.epochs is not None and step >= total_steps
        )
        stop = torch.tensor([float(out_of_time or at_limit)], device=device)
        if distributed:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)  # every rank stops on the same step
        if stop.item():
            break

        if step == SPEED_STEPS[0]:
            speed_start = time.time()
        if total_steps is None and step == SPEED_STEPS[1]:
            seconds = torch.tensor([(time.time() - speed_start) / (SPEED_STEPS[1] - SPEED_STEPS[0])], device=device)
            if distributed:
                dist.broadcast(seconds, 0)
            total_steps = int(args.planned_hours * 3600 / seconds.item())
            say(f"{seconds.item():.2f}s/step -> LR schedule over {total_steps} steps ({args.planned_hours}h planned)")
        schedule_total = total_steps or max(args.warmup * 10, args.max_steps or 0)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * lr_factor(step, args.warmup, schedule_total)

        batch = stream.batch(step)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = ddp([ex.encoded for ex in batch])
        losses = [
            question_loss(z, torch.tensor(t, device=device), args.brier_weight)
            for ex, per_question in zip(batch, logits)
            for z, t in zip(per_question, ex.targets)
        ]
        loss = torch.stack(losses).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        window_loss += loss.item() * len(losses)
        window_questions += len(losses)
        if step % args.log_every == 0:
            elapsed = time.time() - window_start
            say(
                f"step {step} loss {window_loss / window_questions:.4f} lr {optimizer.param_groups[0]['lr']:.2e} "
                f"{elapsed / args.log_every:.2f}s/step trained {(time.time() - train_start) / 3600:.2f}h"
            )
            window_loss, window_questions, window_start = 0.0, 0, time.time()

        # Ranks must agree: deciding on each rank's own clock deadlocked run 2 (one in barrier, one in backward).
        due = torch.tensor([float((time.time() - last_checkpoint) / 60 > args.checkpoint_minutes)], device=device)
        if distributed:
            dist.all_reduce(due, op=dist.ReduceOp.MAX)
        if due.item():
            if rank == 0:
                check_dev()
                save_checkpoint(out_dir / "checkpoint", model, optimizer, scaler, step, total_steps or schedule_total,
                                args, {"best": best, "curve": curve})
                say(f"checkpoint at step {step}")
            if distributed:
                dist.barrier()
            last_checkpoint = time.time()

    if rank == 0:
        if not curve or curve[-1]["step"] != step:
            check_dev()
        save_checkpoint(out_dir / "checkpoint", model, optimizer, scaler, step, total_steps or schedule_total,
                        args, {"best": best, "curve": curve})
        say(f"stopped at step {step} after {(time.time() - train_start) / 3600:.2f}h of training; checkpoint saved")
        # The final evaluation scores the best dev weights, which are also saved
        # on their own (adapter + head only) for Phase 5.
        load_trainable(model, best)
        _save({"lora": best["lora"], "head": best["head"], "step": best["step"], "dev_nll": best["nll"]}, out_dir / "best")
        say(f"evaluating best weights from step {best['step']} (dev nll {best['nll']:.4f})")
        heldout, _ = load_examples(args.data_dir / "heldout.jsonl", tokenizer, args)
        eval_rng.shuffle(heldout)
        result = evaluate(model, dev, _stratified(heldout, args.eval_heldout_per_source), args, out_dir)
        result.update({"step": step, "best_step": best["step"], "dev_curve": curve,
                       "train_hours_this_session": (time.time() - train_start) / 3600, "world": world})
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        say(json.dumps({"step": step, "best_step": best["step"], "temperature": result["temperature"], "gate": result["gate"],
                        "dev_all": result["dev"]["scaled"]["all"], "heldout": {k: v.get("accuracy") for k, v in result["heldout"]["scaled"].items()}}, indent=2))
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
