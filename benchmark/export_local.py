#!/usr/bin/env python3
"""Export the RoboVista dataset to a plain directory: questions.json plus
pre-resized JPEG images (same 720 px / q85 preprocessing as run_benchmark.py).

This lets inference run in an environment without the `datasets` library.

Usage:
    python benchmark/export_local.py --out data_local
"""
import argparse
import json
from pathlib import Path

from run_benchmark import DEFAULT_DATASET, resize_image


def main():
    parser = argparse.ArgumentParser(description="Export RoboVista to a local directory")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--parquet")
    parser.add_argument("--out", default="data_local")
    args = parser.parse_args()

    from datasets import Dataset, load_dataset

    if args.parquet:
        ds = Dataset.from_parquet(args.parquet)
    else:
        ds = load_dataset(args.dataset, split="train")

    out = Path(args.out)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    questions = []
    for i in range(len(ds)):
        row = ds[i]
        qid = row.get("id") or str(i)
        image_files = []
        for j, img in enumerate(row["images"]):
            name = f"{qid}_{j}.jpg"
            (images_dir / name).write_bytes(resize_image(img))
            image_files.append(f"images/{name}")
        questions.append({
            "question_id": qid,
            "question": row["question"],
            "choices": row["choices"],
            "correct_answer": row["correct_answer"],
            "reasoning": row.get("reasoning", ""),
            "domain": (row.get("domain") or "").strip(),
            "task": (row.get("task") or "").strip(),
            "ability_type": (row.get("ability_type") or "").strip(),
            "images": image_files,
        })

    with open(out / "questions.json", "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Exported {len(questions)} questions to {out}/")


if __name__ == "__main__":
    main()
