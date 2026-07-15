#!/usr/bin/env python3
"""Benchmark a local HuggingFace vision-language model on RoboVista.

Runs the model directly through transformers (no API server needed), with
batched greedy decoding. Reads the dataset from a directory produced by
export_local.py, so the inference environment only needs torch, transformers,
and Pillow. Produces the same summary JSON format as run_benchmark.py, with
domain/ability metadata added per question.

Usage:
    python benchmark/export_local.py --out data_local   # once, needs `datasets`
    python benchmark/run_benchmark_local.py \
        --data-dir data_local \
        --model-path /path/to/Qwen2-VL-7B-Instruct \
        --prompts standard cot

    # In-context learning with 2 same-domain exemplars:
    python benchmark/run_benchmark_local.py \
        --data-dir data_local \
        --model-path /path/to/Qwen2-VL-7B-Instruct \
        --prompts standard --icl-k 2
"""
import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_benchmark import CHOICE_LETTERS, format_question_with_choices, parse_answer

SAVE_INTERVAL = 10


def load_questions(data_dir: Path) -> List[Dict[str, Any]]:
    with open(data_dir / "questions.json") as f:
        rows = json.load(f)
    questions = []
    for row in rows:
        choices = {letter: text for letter, text in zip(CHOICE_LETTERS, row["choices"]) if text}
        if not choices:
            continue
        row = dict(row)
        row["choices"] = choices
        questions.append(row)
    return questions


def load_images(data_dir: Path, q: Dict[str, Any]) -> List:
    from PIL import Image

    return [Image.open(data_dir / rel).convert("RGB") for rel in q["images"]]


def image_patches(data_dir: Path, q: Dict[str, Any]) -> int:
    """Vision-tower patch count for a question's images (14 px patches)."""
    from PIL import Image

    total = 0
    for rel in q["images"]:
        with Image.open(data_dir / rel) as im:
            w, h = im.size
        total += math.ceil(w / 14) * math.ceil(h / 14)
    return total


def pick_exemplars(q: Dict[str, Any], questions: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Deterministically pick k same-domain exemplars (never the question
    itself), preferring exemplars with few images to bound context size."""
    pool = [x for x in questions if x["domain"] == q["domain"] and x["question_id"] != q["question_id"]]
    pool.sort(key=lambda x: (len(x["images"]), x["question_id"]))
    pool = pool[: max(4 * k, 8)]
    rng = random.Random(q["question_id"])
    return rng.sample(pool, min(k, len(pool)))


def build_messages(
    data_dir: Path,
    q: Dict[str, Any],
    prompt_config: Dict[str, str],
    exemplars: List[Dict[str, Any]],
    patch_budget: int = 0,
) -> Tuple[List[Dict[str, Any]], List]:
    """Chat messages plus the flat image list, exemplar images first."""
    messages = [{"role": "system", "content": prompt_config["system_prompt"]}]
    images: List = []

    for ex in exemplars:
        ex_images = load_images(data_dir, ex)
        prompt = format_question_with_choices(ex["question"], ex["choices"], prompt_config["question_suffix"])
        messages.append({
            "role": "user",
            "content": [{"type": "image"} for _ in ex_images] + [{"type": "text", "text": prompt}],
        })
        reply = f"{ex['reasoning']} The answer is {ex['correct_answer']}." if ex["reasoning"] else ex["correct_answer"]
        messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        images.extend(ex_images)

    q_images = load_images(data_dir, q)
    prompt = format_question_with_choices(q["question"], q["choices"], prompt_config["question_suffix"])
    messages.append({
        "role": "user",
        "content": [{"type": "image"} for _ in q_images] + [{"type": "text", "text": prompt}],
    })
    images.extend(q_images)

    # If a single prompt exceeds the vision patch budget (many multi-view
    # images), downscale all of its images so the batch cannot OOM.
    if patch_budget:
        total = sum(math.ceil(im.width / 14) * math.ceil(im.height / 14) for im in images)
        if total > patch_budget:
            factor = (patch_budget / total) ** 0.5
            images = [
                im.resize((max(28, int(im.width * factor)), max(28, int(im.height * factor))))
                for im in images
            ]
    return messages, images


def run(args):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    with open(args.prompt_config) as f:
        prompt_configs = json.load(f)
    for name in args.prompts:
        if name not in prompt_configs:
            sys.exit(f"Error: prompt '{name}' not found in {args.prompt_config} (available: {list(prompt_configs)})")

    data_dir = Path(args.data_dir)
    questions = load_questions(data_dir)
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"Loaded {len(questions)} questions from {data_dir}/")

    print(f"Loading model {args.model_path} ...")
    # sdpa produces garbage ("!!!!") for padded sequences in batched Qwen2-VL
    # inference on this stack, so the text decoder needs eager. The vision
    # tower sees no padding (cu_seqlens concatenation), and eager there
    # materializes a full fp32 attention matrix over every image in the batch
    # (OOMs at batch 8), so it stays on sdpa.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
    ).to(args.device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    if args.vision_attn and args.vision_attn != args.attn:
        from transformers.models.qwen2_vl import modeling_qwen2_vl as qwen2_vl
        vision_cls = {
            "eager": qwen2_vl.VisionAttention,
            "sdpa": qwen2_vl.VisionSdpaAttention,
        }[args.vision_attn]
        for blk in model.visual.blocks:
            blk.attn.__class__ = vision_cls
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_key = args.model_key or Path(args.model_path).name

    for prompt_name in args.prompts:
        run_one(args, data_dir, questions, model, processor, prompt_configs[prompt_name], prompt_name, model_key, output_dir)


def run_one(args, data_dir, questions, model, processor, prompt_config, prompt_name, model_key, output_dir):
    import torch

    output_model_name = model_key.replace("/", "-")
    if prompt_name != "standard":
        output_model_name = f"{output_model_name}_{prompt_name}"
    if args.icl_k:
        output_model_name = f"{output_model_name}_icl{args.icl_k}"

    max_new_tokens = args.max_new_tokens or (1024 if prompt_name in ("cot", "rl") else 32)

    intermediate_path = output_dir / f"intermediate_{output_model_name}.json"
    existing: Dict[str, Dict[str, Any]] = {}
    if intermediate_path.exists():
        with open(intermediate_path) as f:
            existing = {r["question_id"]: r for r in json.load(f)}
        print(f"Resuming: {len(existing)} questions already completed.")

    remaining = [q for q in questions if q["question_id"] not in existing]
    all_results = dict(existing)

    def save_intermediate():
        with open(intermediate_path, "w") as f:
            json.dump(list(all_results.values()), f, indent=2)

    # Batch by total vision patches, not question count: the vision tower
    # attends over all batch images as one concatenated sequence, so memory is
    # quadratic in total image area. Questions have 1-8 images each.
    entries = []
    for q in remaining:
        exemplars = pick_exemplars(q, questions, args.icl_k) if args.icl_k else []
        patches = sum(image_patches(data_dir, item) for item in [q] + exemplars)
        entries.append((q, exemplars, min(patches, args.max_batch_patches)))

    batches, cur, cur_patches = [], [], 0
    for q, exemplars, patches in entries:
        if cur and (len(cur) >= args.batch_size or cur_patches + patches > args.max_batch_patches):
            batches.append(cur)
            cur, cur_patches = [], 0
        cur.append((q, exemplars))
        cur_patches += patches
    if cur:
        batches.append(cur)

    pbar = tqdm(total=len(questions), initial=len(existing), desc=output_model_name)
    for batch_index, batch_entries in enumerate(batches):
        batch = [q for q, _ in batch_entries]
        texts, image_lists = [], []
        for q, exemplars in batch_entries:
            messages, images = build_messages(data_dir, q, prompt_config, exemplars, args.max_batch_patches)
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            image_lists.append(images)

        inputs = processor(text=texts, images=image_lists, padding=True, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        responses = processor.batch_decode(trimmed, skip_special_tokens=True)

        for q, response in zip(batch, responses):
            response = response.strip()
            predicted = parse_answer(response)
            all_results[q["question_id"]] = {
                "question_id": q["question_id"],
                "predicted_answer": predicted,
                "model_response": response,
                "correct_answer": q["correct_answer"],
                "is_correct": predicted == q["correct_answer"],
                "domain": q["domain"],
                "ability_type": q["ability_type"],
            }
            pbar.update(1)
        if batch_index % max(1, SAVE_INTERVAL // args.batch_size) == 0:
            save_intermediate()
        # Return cached VRAM between batches; the GPU is shared with other jobs
        # and eager-attention prefill peaks are large.
        del inputs, output_ids, trimmed
        torch.cuda.empty_cache()
    pbar.close()

    results = [all_results[q["question_id"]] for q in questions if q["question_id"] in all_results]
    correct = sum(1 for r in results if r.get("is_correct"))
    accuracy = correct / len(results) if results else 0.0

    summary = {
        "model": output_model_name,
        "model_id": args.model_path,
        "prompt": prompt_name,
        "icl_k": args.icl_k,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_questions": len(results),
        "correct": correct,
        "errors": 0,
        "accuracy": accuracy,
        "results": results,
    }
    summary_path = output_dir / f"summary_{output_model_name}_{summary['timestamp']}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    intermediate_path.unlink(missing_ok=True)

    print(f"\n{output_model_name}: {correct}/{len(results)} correct ({accuracy:.1%})")
    print(f"Saved {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark a local VLM on the RoboVista dataset")
    parser.add_argument("--data-dir", default="data_local", help="Directory produced by export_local.py")
    parser.add_argument("--model-path", required=True, help="Local path or HF id of the model")
    parser.add_argument("--model-key", help="Short name used in output filenames (defaults to model dir name)")
    parser.add_argument("--adapter", help="Optional PEFT adapter path, merged into the base model (needs peft on PYTHONPATH)")
    parser.add_argument("--prompt-config", default=str(Path(__file__).parent / "prompts.json"))
    parser.add_argument("--prompts", nargs="+", default=["standard"], help="Prompt configurations to run")
    parser.add_argument("--icl-k", type=int, default=0, help="Number of same-domain in-context exemplars")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=8, help="Max questions per batch")
    parser.add_argument("--max-batch-patches", type=int, default=14000,
                        help="Max total vision patches (14px) per batch; bounds vision attention memory")
    parser.add_argument("--max-new-tokens", type=int, help="Defaults: 32 (standard), 1024 (cot)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn", default="eager", help="Attention implementation (eager is correct for batched padding; sdpa is not)")
    parser.add_argument("--vision-attn", default="sdpa", help="Attention implementation for the vision tower (sdpa avoids the eager fp32 OOM; no padding there)")
    parser.add_argument("--max-questions", type=int, help="Limit number of questions (for quick tests)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
