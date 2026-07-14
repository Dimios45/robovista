#!/usr/bin/env python3
"""Export a training subset of Robo2VLM-1 (keplerccc/Robo2VLM-1, 684k robot
VQA) into the same local directory format as benchmark/export_local.py, for
GRPO/SFT training with RoboVista kept as a pure held-out benchmark.

Streams the dataset (107 GB total) with a shuffle buffer so nothing large is
stored; writes resized JPEGs plus questions.json.

Usage:
    python rl/export_robo2vlm.py --train 5000 --heldout 500 --out rl_data
"""
import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))
from run_benchmark import CHOICE_LETTERS, resize_image


def main():
    parser = argparse.ArgumentParser(description="Export a Robo2VLM-1 subset")
    parser.add_argument("--dataset", default="keplerccc/Robo2VLM-1")
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--heldout", type=int, default=500)
    parser.add_argument("--buffer", type=int, default=20000, help="Streaming shuffle buffer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="rl_data")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train", streaming=True)
    ds = ds.shuffle(buffer_size=args.buffer, seed=args.seed)

    splits = [("train", args.train), ("heldout", args.heldout)]
    for split, _ in splits:
        (Path(args.out) / split / "images").mkdir(parents=True, exist_ok=True)

    it = iter(ds)
    seen_ids = set()
    for split, count in splits:
        out = Path(args.out) / split
        questions = []
        while len(questions) < count:
            row = next(it)
            qid = row["id"]
            if qid in seen_ids:
                continue
            seen_ids.add(qid)

            choices = row["choices"]
            if isinstance(choices, str):
                choices = ast.literal_eval(choices)
            answer_index = int(row["correct_answer"])
            if not 0 <= answer_index < len(choices) or len(choices) > len(CHOICE_LETTERS):
                continue

            name = f"{qid}.jpg"
            (out / "images" / name).write_bytes(resize_image(row["image"]))
            questions.append({
                "question_id": qid,
                "question": row["question"],
                "choices": [str(c) for c in choices],
                "correct_answer": CHOICE_LETTERS[answer_index],
                "reasoning": "",
                "domain": "robo2vlm",
                "task": "",
                "ability_type": "",
                "images": [f"images/{name}"],
            })
            if len(questions) % 500 == 0:
                print(f"{split}: {len(questions)}/{count}", flush=True)

        with open(out / "questions.json", "w") as f:
            json.dump(questions, f, indent=2)
        print(f"Exported {len(questions)} questions to {out}/", flush=True)


if __name__ == "__main__":
    main()
