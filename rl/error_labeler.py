#!/usr/bin/env python3
"""Human error-taxonomy labeler for the RoboVista-R1 mini-paper.

Samples ~100 GRPO-think errors on RoboVista (stratified by ability type),
shows each with the model's reasoning next to the expert's, and records one of
four failure labels. Labels persist to rl_runs/error_labels.json (resume-safe).

Usage:
    python rl/error_labeler.py            # http://localhost:7861
"""
import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).parent.parent
LABELS = [
    "Misidentification (saw the wrong thing)",
    "Spatial reasoning (right objects, wrong geometry)",
    "Task reasoning (right scene, wrong plan/logic)",
    "Format / refusal / other",
]
LABELS_PATH = ROOT / "rl_runs" / "error_labels.json"


def load_errors(summary_glob, per_stratum, seed):
    paths = sorted(glob.glob(str(ROOT / "results" / summary_glob)))
    summary = json.load(open(paths[-1]))
    with open(ROOT / "data_local" / "questions.json") as f:
        questions = {q["question_id"]: q for q in json.load(f)}
    errors = [r for r in summary["results"] if not r.get("is_correct")]
    strata = defaultdict(list)
    for r in errors:
        strata[r.get("ability_type") or "unknown"].append(r)
    rng = random.Random(seed)
    sample = []
    for ab, rows in sorted(strata.items()):
        rng.shuffle(rows)
        sample.extend(rows[:per_stratum])
    rng.shuffle(sample)
    return [(r, questions[r["question_id"]]) for r in sample if r["question_id"] in questions]


def main():
    parser = argparse.ArgumentParser(description="Error taxonomy labeler")
    parser.add_argument("--summary-glob", default="summary_qwen2vl-7b-grpo_rl_*.json")
    parser.add_argument("--per-stratum", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    items = load_errors(args.summary_glob, args.per_stratum, args.seed)
    labels = {}
    if LABELS_PATH.exists():
        labels = json.load(open(LABELS_PATH))
    print(f"{len(items)} errors to label; {len(labels)} already labeled.")

    def first_unlabeled():
        for i, (r, _) in enumerate(items):
            if r["question_id"] not in labels:
                return i
        return len(items) - 1

    def render(i):
        i = max(0, min(i, len(items) - 1))
        r, q = items[i]
        imgs = [str(ROOT / "data_local" / rel) for rel in q["images"]]
        choice_lines = []
        for letter, text in q["choices"].items() if isinstance(q["choices"], dict) else zip("ABCDE", q["choices"]):
            if not text:
                continue
            mark = " ✅" if letter == r["correct_answer"] else (" ❌ (model)" if letter == r["predicted_answer"] else "")
            choice_lines.append(f"**{letter}.** {text}{mark}")
        done = len(labels)
        head = (f"### {done}/{len(items)} labeled — item {i + 1}\n"
                f"`{r['question_id']}` · domain **{q['domain']}** · ability **{r.get('ability_type')}**")
        body = (f"**Question:** {q['question']}\n\n" + "\n\n".join(choice_lines))
        model_md = f"**Model answered {r['predicted_answer']}:**\n\n> {r['model_response'][:2000]}"
        expert_md = f"**Expert reasoning (answer {r['correct_answer']}):**\n\n> {q['reasoning']}"
        current = labels.get(r["question_id"], {}).get("label", "(unlabeled)")
        return imgs, head, body, model_md, expert_md, f"Current label: **{current}**", i

    def label_and_next(i, label):
        r, _ = items[i]
        labels[r["question_id"]] = {"label": label, "ability_type": r.get("ability_type")}
        LABELS_PATH.parent.mkdir(exist_ok=True)
        with open(LABELS_PATH, "w") as f:
            json.dump(labels, f, indent=2)
        nxt = i + 1
        for j in range(i + 1, len(items)):
            if items[j][0]["question_id"] not in labels:
                nxt = j
                break
        return render(nxt)

    with gr.Blocks(title="RoboVista-R1 error labeler") as demo:
        gr.Markdown("## GRPO-think error labeling — pick the *primary* failure cause")
        idx = gr.State(first_unlabeled())
        head = gr.Markdown()
        with gr.Row():
            gallery = gr.Gallery(columns=2, height=360, label="Images")
            with gr.Column():
                body = gr.Markdown()
        with gr.Row():
            model_md = gr.Markdown()
            expert_md = gr.Markdown()
        status = gr.Markdown()
        with gr.Row():
            btns = [gr.Button(l, variant="primary" if i == 0 else "secondary") for i, l in enumerate(LABELS)]
        with gr.Row():
            prev_b = gr.Button("← Prev", size="sm")
            next_b = gr.Button("Next →", size="sm")

        outs = [gallery, head, body, model_md, expert_md, status, idx]
        for b, l in zip(btns, LABELS):
            b.click(lambda i, l=l: label_and_next(i, l), inputs=[idx], outputs=outs)
        prev_b.click(lambda i: render(i - 1), inputs=[idx], outputs=outs)
        next_b.click(lambda i: render(i + 1), inputs=[idx], outputs=outs)
        demo.load(lambda i: render(i), inputs=[idx], outputs=outs)

    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
