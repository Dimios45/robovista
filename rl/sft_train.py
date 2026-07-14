#!/usr/bin/env python3
"""LoRA supervised fine-tuning baseline on the same Robo2VLM-1 subset used by
GRPO (rl/grpo_train.py), for the RL-vs-SFT ablation. Targets are the correct
answer letter under the benchmark's `standard` prompt; loss is computed on
answer tokens only.

Usage:
    PYTHONPATH=.rl-deps python rl/sft_train.py \
        --data-dir rl_data/train \
        --model-path /path/to/Qwen2-VL-7B-Instruct \
        --output-dir rl_runs/sft
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))
from run_benchmark import format_question_with_choices
from run_benchmark_local import load_images, load_questions

SYSTEM_PROMPT = (
    "You are a helpful assistant analyzing robotic manipulation images. Answer "
    "the multiple choice question based on the images. Respond with only the "
    "letter of your answer."
)
QUESTION_SUFFIX = "Answer with the letter only (A, B, C, D, or E)."


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT baseline on robot VQA")
    parser.add_argument("--data-dir", default="rl_data/train")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="rl_runs/sft")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--max-steps", type=int, help="Stop after N steps (smoke tests)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    questions = load_questions(data_dir)
    print(f"Loaded {len(questions)} training questions from {data_dir}/", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(args.device)
    # The frozen vision tower is safe on sdpa (no padding there) and much
    # faster/leaner than eager; text stays eager (see run_benchmark_local.py).
    from transformers.models.qwen2_vl import modeling_qwen2_vl as qwen2_vl

    for blk in model.visual.blocks:
        blk.attn.__class__ = qwen2_vl.VisionSdpaAttention
    processor = AutoProcessor.from_pretrained(args.model_path)
    # Right padding so the answer tokens sit at a fixed offset from the start;
    # label masking below assumes it. (Left padding is only needed for
    # generation, which SFT never does.)
    processor.tokenizer.padding_side = "right"

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(questions) / args.batch_size) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / args.warmup_steps)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))),
    )

    def encode_batch(batch):
        prompt_texts, full_texts, image_lists = [], [], []
        for q in batch:
            prompt = format_question_with_choices(q["question"], q["choices"], QUESTION_SUFFIX)
            images = load_images(data_dir, q)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]},
            ]
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_texts.append(prompt_text)
            full_texts.append(prompt_text + q["correct_answer"] + "<|im_end|>")
            image_lists.append(images)

        inputs = processor(text=full_texts, images=image_lists, padding=True, return_tensors="pt")
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        # Mask out everything except the answer tokens (answer letter + im_end).
        for i, prompt_text in enumerate(prompt_texts):
            prompt_len = len(processor(text=[prompt_text], images=[image_lists[i]])["input_ids"][0])
            labels[i, :prompt_len] = -100
        inputs["labels"] = labels
        return inputs.to(args.device)

    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(questions)))
        random.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            t0 = time.time()
            batch = [questions[i] for i in order[start : start + args.batch_size]]
            inputs = encode_batch(batch)
            loss = model(**inputs, use_cache=False).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 10 == 0:
                record = {
                    "step": step, "epoch": epoch, "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0], "seconds": round(time.time() - t0, 2),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if step % args.save_every == 0:
                model.save_pretrained(str(output_dir / "adapter_latest"))
            del inputs, loss
            if step % 20 == 0:
                torch.cuda.empty_cache()
            if args.max_steps and step >= args.max_steps:
                print(f"Stopping at --max-steps {args.max_steps}", flush=True)
                model.save_pretrained(str(output_dir / "adapter_latest"))
                return

    model.save_pretrained(str(output_dir / "adapter_latest"))
    model.save_pretrained(str(output_dir / "adapter_final"))
    print("SFT complete.", flush=True)


if __name__ == "__main__":
    main()
