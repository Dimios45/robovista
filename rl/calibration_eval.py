#!/usr/bin/env python3
"""Calibration evaluation: A-E answer-letter probabilities from a single
forward pass (no generation), then ECE and reliability data.

For each question, the model sees the benchmark's `standard` prompt and we read
the next-token logits for the letter tokens A-E (restricted softmax). The
argmax letter is the prediction; its probability is the confidence.

Outputs results/calibration_<key>.json with per-question records and summary
ECE / accuracy.

Usage:
    PYTHONPATH=.rl-deps python rl/calibration_eval.py \
        --data-dir data_local --model-path <model> [--adapter <path>] --model-key base
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))
from run_benchmark import CHOICE_LETTERS, format_question_with_choices
from run_benchmark_local import image_patches, load_images, load_questions

SYSTEM_PROMPT = (
    "You are a helpful assistant analyzing robotic manipulation images. Answer "
    "the multiple choice question based on the images. Respond with only the "
    "letter of your answer."
)
SUFFIX = "Answer with the letter only (A, B, C, D, or E)."
MAX_BATCH_PATCHES = 14000


def ece(records, bins=15):
    total = len(records)
    if not total:
        return float("nan")
    acc_err = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [r for r in records if lo < r["confidence"] <= hi]
        if not sel:
            continue
        conf = sum(r["confidence"] for r in sel) / len(sel)
        acc = sum(r["is_correct"] for r in sel) / len(sel)
        acc_err += abs(conf - acc) * len(sel) / total
    return acc_err


def main():
    parser = argparse.ArgumentParser(description="A-E logprob calibration eval")
    parser.add_argument("--data-dir", default="data_local")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-patches", type=int, default=MAX_BATCH_PATCHES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from transformers.models.qwen2_vl import modeling_qwen2_vl as qwen2_vl

    data_dir = Path(args.data_dir)
    questions = load_questions(data_dir)
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"{len(questions)} questions from {data_dir}/", flush=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(args.device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    for blk in model.visual.blocks:
        blk.attn.__class__ = qwen2_vl.VisionSdpaAttention
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"  # aligns final position across the batch

    letter_ids = [processor.tokenizer(L, add_special_tokens=False)["input_ids"][0]
                  for L in CHOICE_LETTERS]

    # patch-budget batching, same rationale as the benchmark runner
    entries = [(q, min(image_patches(data_dir, q), args.max_batch_patches)) for q in questions]
    batches, cur, cur_p = [], [], 0
    for q, p in entries:
        if cur and (len(cur) >= args.batch_size or cur_p + p > args.max_batch_patches):
            batches.append(cur)
            cur, cur_p = [], 0
        cur.append(q)
        cur_p += p
    if cur:
        batches.append(cur)

    records = []
    for batch in tqdm(batches, desc=args.model_key):
        texts, image_lists = [], []
        for q in batch:
            prompt = format_question_with_choices(q["question"], q["choices"], SUFFIX)
            images = load_images(data_dir, q)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]},
            ]
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            image_lists.append(images)
        inputs = processor(text=texts, images=image_lists, padding=True, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[:, -1, :]
        for q, row in zip(batch, logits):
            offered = [L for L in q["choices"]]  # letters actually offered
            ids = [letter_ids[CHOICE_LETTERS.index(L)] for L in offered]
            probs = torch.softmax(row[ids].float(), dim=-1)
            k = int(probs.argmax())
            records.append({
                "question_id": q["question_id"],
                "predicted_answer": offered[k],
                "confidence": float(probs[k]),
                "probs": {L: float(p) for L, p in zip(offered, probs)},
                "correct_answer": q["correct_answer"],
                "is_correct": offered[k] == q["correct_answer"],
                "domain": q["domain"],
                "ability_type": q["ability_type"],
            })
        del inputs, logits
        torch.cuda.empty_cache()

    acc = sum(r["is_correct"] for r in records) / len(records)
    summary = {
        "model_key": args.model_key, "data_dir": str(data_dir),
        "n": len(records), "accuracy": acc, "ece15": ece(records),
        "mean_confidence": sum(r["confidence"] for r in records) / len(records),
        "records": records,
    }
    out = Path(args.output_dir) / f"calibration_{args.model_key}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{args.model_key}: acc {acc:.1%} | ECE15 {summary['ece15']:.3f} | "
          f"mean conf {summary['mean_confidence']:.1%}\nSaved {out}")


if __name__ == "__main__":
    main()
