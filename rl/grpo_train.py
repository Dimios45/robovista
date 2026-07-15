#!/usr/bin/env python3
"""GRPO (Group Relative Policy Optimization) fine-tuning of Qwen2-VL on
multiple-choice robot VQA with verifiable rewards — hand-written loop, no TRL
(this stack is Python 3.9 / transformers 4.48 / torch 2.5.1+rocm6.2).

Per step: P prompts x G sampled completions each. Reward = +1.0 for the
correct letter, +0.2 for well-formed <think>...</think> 'Answer: X' output.
Advantages are group-normalized (no value model). Loss is token-level policy
gradient with a k3 KL penalty against the reference policy — the same model
with the LoRA adapter disabled, so no second copy in memory.

Usage:
    PYTHONPATH=.rl-deps python rl/grpo_train.py \
        --data-dir rl_data/train \
        --model-path /path/to/Qwen2-VL-7B-Instruct \
        --output-dir rl_runs/grpo
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))
from run_benchmark import format_question_with_choices, parse_answer
from run_benchmark_local import load_images, load_questions

SYSTEM_PROMPT = "You are a helpful assistant analyzing robotic manipulation images."
QUESTION_SUFFIX = (
    "Think step by step inside <think> </think> tags, then give your final "
    "answer on a new line as 'Answer: X' where X is the letter of your choice."
)
FORMAT_RE = re.compile(r"^\s*<think>.*?</think>\s*Answer:\s*([A-E])\s*$", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"Answer:\s*([A-E])\b", re.IGNORECASE)


def compute_reward(text: str, correct: str):
    """Returns (reward, is_correct, is_formatted)."""
    fmt = FORMAT_RE.match(text)
    m = fmt or ANSWER_RE.search(text)
    predicted = m.group(1).upper() if m else parse_answer(text)
    is_correct = predicted == correct
    reward = (1.0 if is_correct else 0.0) + (0.2 if fmt else 0.0)
    return reward, is_correct, bool(fmt)


def completion_mask_from_ids(completion_ids, eos_token_id):
    """Mask covering tokens up to and including the first EOS (im_end)."""
    import torch

    is_eos = completion_ids == eos_token_id
    # Positions strictly after the first EOS get masked out.
    after_eos = is_eos.cumsum(dim=1) - is_eos.int() > 0
    return (~after_eos).to(torch.float32)


def selective_log_softmax(logits, index):
    """Per-token log-probs of `index` under `logits` without materializing the
    full log_softmax for longer than one row (memory)."""
    import torch

    logps = []
    for row_logits, row_index in zip(logits, index):
        row_logps = torch.log_softmax(row_logits.float(), dim=-1)
        logps.append(torch.gather(row_logps, 1, row_index.unsqueeze(-1)).squeeze(-1))
    return torch.stack(logps)


def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning on robot VQA")
    parser.add_argument("--data-dir", default="rl_data/train")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="rl_runs/grpo")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--kl-beta", type=float, default=0.04)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    questions = load_questions(data_dir)
    print(f"Loaded {len(questions)} training questions from {data_dir}/", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    state_path = output_dir / "trainer_state.pt"

    print(f"Loading model {args.model_path} ...", flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)
    eos_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    start_step = 0
    latest = output_dir / "adapter_latest"
    if latest.exists() and state_path.exists():
        model = PeftModel.from_pretrained(model, str(latest), is_trainable=True)
        state = torch.load(state_path, weights_only=False)
        start_step = state["step"]
        print(f"Resuming from step {start_step}", flush=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        state = None
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / args.warmup_steps)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, args.steps)))),
    )
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    rng = random.Random(args.seed + start_step)
    order = list(range(len(questions)))
    rng.shuffle(order)
    cursor = (start_step * args.prompts_per_step) % len(order)

    def next_question():
        nonlocal cursor
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
        q = questions[order[cursor]]
        cursor += 1
        return q

    def build_prompt(q: Dict[str, Any]):
        prompt = format_question_with_choices(q["question"], q["choices"], QUESTION_SUFFIX)
        images = load_images(data_dir, q)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text, images

    for step in range(start_step, args.steps):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        stats = {"reward": [], "correct": [], "formatted": [], "kl": [], "loss": [], "skipped_groups": 0}

        # Resample degenerate groups (zero reward variance = no signal) so
        # every step backprops through prompts_per_step useful groups.
        useful, attempts = 0, 0
        while useful < args.prompts_per_step and attempts < 3 * args.prompts_per_step:
            attempts += 1
            q = next_question()
            text, images = build_prompt(q)
            G = args.group_size
            inputs = processor(text=[text] * G, images=[images] * G, return_tensors="pt").to(args.device)
            prompt_len = inputs["input_ids"].shape[1]

            model.eval()
            with torch.no_grad():
                # top_k=0 disables the top_k=1 (greedy!) that Qwen2-VL ships
                # in its generation_config — without this, all G completions
                # are identical and every group is degenerate.
                out = model.generate(
                    **inputs, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    top_k=0, max_new_tokens=args.max_new_tokens, use_cache=True,
                )
            completion_ids = out[:, prompt_len:]
            completions = processor.batch_decode(completion_ids, skip_special_tokens=True)

            rewards, corrects, fmts = [], [], []
            for c in completions:
                r, ok, fmt = compute_reward(c.strip(), q["correct_answer"])
                rewards.append(r)
                corrects.append(ok)
                fmts.append(fmt)
            stats["reward"].extend(rewards)
            stats["correct"].extend(corrects)
            stats["formatted"].extend(fmts)

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=args.device)
            if rewards_t.std() < 1e-6:
                stats["skipped_groups"] += 1
                continue
            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-4)

            full_ids = torch.cat([inputs["input_ids"], completion_ids], dim=1)
            attn = torch.cat(
                [inputs["attention_mask"], (completion_ids != processor.tokenizer.pad_token_id).long()], dim=1,
            )
            comp_mask = completion_mask_from_ids(completion_ids, eos_id)

            model.train()
            fwd = dict(
                input_ids=full_ids, attention_mask=attn,
                pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
                use_cache=False,
            )
            logits = model(**fwd).logits[:, prompt_len - 1 : -1]
            logps = selective_log_softmax(logits, completion_ids)
            del logits

            with torch.no_grad(), model.disable_adapter():
                ref_logits = model(**fwd).logits[:, prompt_len - 1 : -1]
                ref_logps = selective_log_softmax(ref_logits, completion_ids)
                del ref_logits

            # k3 KL estimator, per token.
            kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
            per_token = -advantages.unsqueeze(1) * logps + args.kl_beta * kl
            denom = comp_mask.sum().clamp(min=1.0)
            loss = (per_token * comp_mask).sum() / denom / args.prompts_per_step
            loss.backward()

            stats["kl"].append(((kl.detach() * comp_mask).sum() / denom).item())
            stats["loss"].append(loss.item() * args.prompts_per_step)
            useful += 1
            del logps, ref_logps, kl, per_token, inputs, full_ids, attn, out
            torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        n = max(1, len(stats["reward"]))
        record = {
            "step": step + 1,
            "reward_mean": sum(stats["reward"]) / n,
            "accuracy": sum(stats["correct"]) / n,
            "format_rate": sum(stats["formatted"]) / n,
            "kl": sum(stats["kl"]) / max(1, len(stats["kl"])),
            "loss": sum(stats["loss"]) / max(1, len(stats["loss"])),
            "skipped_groups": stats["skipped_groups"],
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - t0, 1),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            model.save_pretrained(str(output_dir / "adapter_latest"))
            model.save_pretrained(str(output_dir / f"adapter_step{step + 1:04d}"))
            torch.save(
                {"step": step + 1, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                state_path,
            )
            # Keep only the two most recent step checkpoints.
            snaps = sorted(output_dir.glob("adapter_step*"))
            for old in snaps[:-2]:
                for p in sorted(old.rglob("*"), reverse=True):
                    p.unlink() if p.is_file() else p.rmdir()
                old.rmdir()

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
