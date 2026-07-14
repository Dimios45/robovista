#!/usr/bin/env python3
"""Analyze RoboVista benchmark summaries: per-domain and per-ability accuracy,
plus pairwise comparisons between runs (e.g. CoT vs standard, ICL vs standard).

Usage:
    python benchmark/analyze_results.py results/summary_*.json
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

DEFAULT_DATASET = "sy-xie/robovista"


def load_metadata(dataset_id: str, parquet: str, data_dir: str):
    """Map question_id -> domain/ability. Prefers a local export_local.py
    directory (no `datasets` dependency); falls back to the Hub dataset."""
    if data_dir:
        with open(Path(data_dir) / "questions.json") as f:
            rows = json.load(f)
        return {
            r["question_id"]: {"domain": r["domain"], "ability_type": r["ability_type"]}
            for r in rows
        }

    from datasets import Dataset, load_dataset

    if parquet:
        ds = Dataset.from_parquet(parquet)
    else:
        ds = load_dataset(dataset_id, split="train")
    meta = ds.remove_columns([c for c in ds.column_names if c == "images"])
    return {
        row["id"]: {
            "domain": (row.get("domain") or "").strip(),
            "ability_type": (row.get("ability_type") or "").strip(),
        }
        for row in meta
    }


def accuracy(results):
    if not results:
        return float("nan"), 0
    return sum(1 for r in results if r.get("is_correct")) / len(results), len(results)


def breakdown(results, key):
    groups = defaultdict(list)
    for r in results:
        groups[r.get(key) or "unknown"].append(r)
    return {g: accuracy(rs) for g, rs in sorted(groups.items())}


def print_table(title, rows):
    print(f"\n{title}")
    width = max(len(name) for name, _ in rows) + 2
    for name, cell in rows:
        print(f"  {name:<{width}} {cell}")


def main():
    parser = argparse.ArgumentParser(description="Analyze RoboVista benchmark summary files")
    parser.add_argument("summaries", nargs="+", help="summary_*.json files")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--parquet")
    parser.add_argument("--data-dir", help="Directory produced by export_local.py (avoids `datasets` dependency)")
    args = parser.parse_args()

    runs, needs_meta = {}, False
    for path in args.summaries:
        with open(path) as f:
            summary = json.load(f)
        results = summary["results"]
        needs_meta = needs_meta or any("domain" not in r for r in results)
        runs[summary["model"]] = results

    if needs_meta:
        meta = load_metadata(args.dataset, args.parquet, args.data_dir)
        for results in runs.values():
            for r in results:
                m = meta.get(r["question_id"], {})
                r.setdefault("domain", m.get("domain"))
                r.setdefault("ability_type", m.get("ability_type"))

    for name, results in runs.items():
        acc, n = accuracy(results)
        unanswered = sum(1 for r in results if r.get("predicted_answer") is None)
        print(f"\n{'=' * 70}\n{name}: {acc:.1%} overall ({n} questions, {unanswered} unparseable)")
        for key, label in [("domain", "By domain"), ("ability_type", "By ability type")]:
            rows = [(g, f"{a:.1%}  (n={n_g})") for g, (a, n_g) in breakdown(results, key).items()]
            print_table(label, rows)

    for (name_a, res_a), (name_b, res_b) in combinations(runs.items(), 2):
        by_id_a = {r["question_id"]: r for r in res_a}
        by_id_b = {r["question_id"]: r for r in res_b}
        common = sorted(set(by_id_a) & set(by_id_b))
        if not common:
            continue
        print(f"\n{'=' * 70}\nDelta: {name_b} minus {name_a} (on {len(common)} common questions)")
        groups = defaultdict(lambda: [0, 0, 0])  # n, correct_a, correct_b
        for qid in common:
            for key in ["domain", "ability_type"]:
                g = f"{key}:{by_id_a[qid].get(key) or 'unknown'}"
                groups[g][0] += 1
                groups[g][1] += bool(by_id_a[qid].get("is_correct"))
                groups[g][2] += bool(by_id_b[qid].get("is_correct"))
        overall = [len(common),
                   sum(bool(by_id_a[q].get("is_correct")) for q in common),
                   sum(bool(by_id_b[q].get("is_correct")) for q in common)]
        rows = [("OVERALL", f"{overall[1]/overall[0]:+.1%} -> {overall[2]/overall[0]:.1%}  (delta {(overall[2]-overall[1])/overall[0]:+.1%})")]
        for g, (n, ca, cb) in sorted(groups.items()):
            rows.append((g, f"{ca/n:.1%} -> {cb/n:.1%}  (delta {(cb-ca)/n:+.1%}, n={n})"))
        print_table("", rows)


if __name__ == "__main__":
    main()
