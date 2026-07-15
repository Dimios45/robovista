#!/usr/bin/env python3
"""Mechanical (judge-free) error analysis for the RoboVista-R1 mini-paper.

Produces rl_runs/error_analysis.json + .md with:
- flip matrices (right->wrong "unlearned" vs wrong->right "learned") per model
  pair, overall and per domain — the mechanism behind SFT's domain collapse
- answer-letter distribution shift per model vs base (chi-square statistic)
- think-trace length vs correctness for think-format runs, by ability type

Usage:
    python rl/error_analysis.py
"""
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent

PAIRS = [  # (name, summary model key A, summary model key B, eval set label)
    ("base->SFT (RoboVista)", "Qwen2-VL-7B-Instruct", "qwen2vl-7b-sft", "robovista"),
    ("base->GRPO (RoboVista)", "Qwen2-VL-7B-Instruct", "qwen2vl-7b-grpo", "robovista"),
    ("base->GRPO-think (RoboVista)", "Qwen2-VL-7B-Instruct", "qwen2vl-7b-grpo_rl", "robovista"),
    ("base->SFT (heldout)", "qwen2vl-7b-base-heldout", "qwen2vl-7b-sft-heldout", "heldout"),
    ("base->GRPO (heldout)", "qwen2vl-7b-base-heldout", "qwen2vl-7b-grpo-heldout", "heldout"),
]
THINK_RUNS = [
    ("Base think (RoboVista)", "qwen2vl-7b-base_rl"),
    ("GRPO think (RoboVista)", "qwen2vl-7b-grpo_rl"),
]
LETTER_RUNS = [
    ("robovista", ["Qwen2-VL-7B-Instruct", "qwen2vl-7b-sft", "qwen2vl-7b-grpo"]),
    ("heldout", ["qwen2vl-7b-base-heldout", "qwen2vl-7b-sft-heldout", "qwen2vl-7b-grpo-heldout"]),
]


def load(model_key):
    paths = sorted(p for p in glob.glob(str(ROOT / "results" / "summary_*.json"))
                   if json.load(open(p))["model"] == model_key)
    if not paths:
        return None
    return {r["question_id"]: r for r in json.load(open(paths[-1]))["results"]}


def chi2(observed_a, observed_b):
    """Chi-square statistic of B's letter distribution against A's proportions."""
    letters = sorted(set(observed_a) | set(observed_b))
    total_a, total_b = sum(observed_a.values()), sum(observed_b.values())
    stat = 0.0
    for L in letters:
        expected = observed_a.get(L, 0) / total_a * total_b if total_a else 0
        if expected > 0:
            stat += (observed_b.get(L, 0) - expected) ** 2 / expected
    return stat, len(letters) - 1


def main():
    report = {"flips": [], "letter_shift": [], "think_length": []}
    md = ["# Mechanical error analysis", ""]

    md += ["## Flip matrices (what changed vs base)", "",
           "| Pair | n | unlearned (right→wrong) | learned (wrong→right) | net | worst domain (unlearned) |",
           "|---|---|---|---|---|---|"]
    for name, ka, kb, _ in PAIRS:
        A, B = load(ka), load(kb)
        if not A or not B:
            continue
        common = sorted(set(A) & set(B))
        per_dom = defaultdict(lambda: [0, 0])
        unlearned = learned = 0
        for q in common:
            a_ok, b_ok = A[q].get("is_correct"), B[q].get("is_correct")
            dom = A[q].get("domain") or "unknown"
            if a_ok and not b_ok:
                unlearned += 1
                per_dom[dom][0] += 1
            elif b_ok and not a_ok:
                learned += 1
                per_dom[dom][1] += 1
        worst = max(per_dom.items(), key=lambda kv: kv[1][0], default=("-", [0, 0]))
        entry = {"pair": name, "n": len(common), "unlearned": unlearned, "learned": learned,
                 "per_domain": {d: {"unlearned": u, "learned": l} for d, (u, l) in sorted(per_dom.items())}}
        report["flips"].append(entry)
        md.append(f"| {name} | {len(common)} | {unlearned} | {learned} | {learned - unlearned:+d} "
                  f"| {worst[0]} ({worst[1][0]}) |")

    md += ["", "## Answer-letter distribution shift vs base (chi-square)", "",
           "| Eval set | Model | A/B/C/D/E/None | chi2 vs base (df) |", "|---|---|---|---|"]
    for eval_set, keys in LETTER_RUNS:
        dists = {}
        for k in keys:
            rows = load(k)
            if rows:
                dists[k] = Counter((r.get("predicted_answer") or "None") for r in rows.values())
        base_key = keys[0]
        for k, dist in dists.items():
            s = "/".join(str(dist.get(L, 0)) for L in ["A", "B", "C", "D", "E", "None"])
            if k == base_key:
                cell = "—"
            else:
                stat, df = chi2(dists[base_key], dist)
                cell = f"{stat:.1f} (df={df})"
            report["letter_shift"].append({"eval_set": eval_set, "model": k,
                                           "dist": dict(dist), "chi2_vs_base": cell})
            md.append(f"| {eval_set} | {k} | {s} | {cell} |")

    md += ["", "## Think-trace length vs correctness (by ability type)", "",
           "| Run | Ability | n | median len (correct) | median len (wrong) |", "|---|---|---|---|---|"]
    for name, key in THINK_RUNS:
        rows = load(key)
        if not rows:
            continue
        groups = defaultdict(lambda: {"c": [], "w": []})
        for r in rows.values():
            ab = r.get("ability_type") or "unknown"
            L = len(r.get("model_response") or "")
            groups[ab]["c" if r.get("is_correct") else "w"].append(L)
        for ab, g in sorted(groups.items()):
            if len(g["c"]) + len(g["w"]) < 20:
                continue
            med = lambda v: sorted(v)[len(v) // 2] if v else 0
            report["think_length"].append({"run": name, "ability": ab,
                                           "n": len(g["c"]) + len(g["w"]),
                                           "median_correct": med(g["c"]),
                                           "median_wrong": med(g["w"])})
            md.append(f"| {name} | {ab} | {len(g['c']) + len(g['w'])} | {med(g['c'])} | {med(g['w'])} |")

    out = ROOT / "rl_runs"
    with open(out / "error_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out / "error_analysis.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
