#!/usr/bin/env python3
"""Statistics engine for the RoboVista-R1 mini-paper.

Consumes benchmark summary JSONs (results/summary_*.json), produces:
- bootstrap 95% CIs for every run's accuracy (overall, per-domain, per-ability)
- exact McNemar tests for pre-registered paired comparisons, per seed and pooled
  (pooled = discordant pairs summed across seeds)
- Holm correction across the per-domain family for the SFT-collapse claim
- seed aggregation (mean +/- sd) tables

Outputs rl_runs/stats_report.json and rl_runs/stats_report.md.

Usage:
    python rl/stats.py                  # auto-discovers results/summary_*.json
    python rl/stats.py --boot 20000
"""
import argparse
import glob
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent

# run key -> (display name, {seed: model_key in summary filenames})
RUNS = {
    "base": ("Base letter-only (RoboVista)", {0: "Qwen2-VL-7B-Instruct"}),
    "base_cot": ("Base CoT (RoboVista)", {0: "Qwen2-VL-7B-Instruct_cot"}),
    "base_rl": ("Base think (RoboVista)", {0: "qwen2vl-7b-base_rl"}),
    "base_icl": ("Base ICL k=2 (RoboVista)", {0: "Qwen2-VL-7B-Instruct_icl2"}),
    "sft": ("SFT letter-only (RoboVista)", {0: "qwen2vl-7b-sft", 1: "qwen2vl-7b-sft-s1", 2: "qwen2vl-7b-sft-s2"}),
    "grpo": ("GRPO letter-only (RoboVista)", {0: "qwen2vl-7b-grpo", 1: "qwen2vl-7b-grpo-s1", 2: "qwen2vl-7b-grpo-s2"}),
    "grpo_rl": ("GRPO think (RoboVista)", {0: "qwen2vl-7b-grpo_rl", 1: "qwen2vl-7b-grpo-s1_rl", 2: "qwen2vl-7b-grpo-s2_rl"}),
    "base_h": ("Base letter-only (heldout)", {0: "qwen2vl-7b-base-heldout"}),
    "sft_h": ("SFT letter-only (heldout)", {0: "qwen2vl-7b-sft-heldout", 1: "qwen2vl-7b-sft-s1-heldout", 2: "qwen2vl-7b-sft-s2-heldout"}),
    "grpo_h": ("GRPO letter-only (heldout)", {0: "qwen2vl-7b-grpo-heldout", 1: "qwen2vl-7b-grpo-s1-heldout", 2: "qwen2vl-7b-grpo-s2-heldout"}),
    "grpo_h_rl": ("GRPO think (heldout)", {0: "qwen2vl-7b-grpo-heldout_rl", 1: "qwen2vl-7b-grpo-s1-heldout_rl", 2: "qwen2vl-7b-grpo-s2-heldout_rl"}),
}

# Pre-registered comparisons: (name, run_a, run_b, subset_key, subset_value)
# Tests whether run_b differs from run_a on paired questions.
COMPARISONS = [
    ("C1 SFT vs base, RoboVista overall", "base", "sft", None, None),
    ("GRPO vs base, RoboVista overall", "base", "grpo", None, None),
    ("GRPO vs SFT, RoboVista overall", "sft", "grpo", None, None),
    ("C2 GRPO-think vs base, high-level decision making", "base", "grpo_rl", "ability_type", "high_level_decision_making"),
    ("C2 GRPO-think vs base, low-level motion awareness", "base", "grpo_rl", "ability_type", "low_level_motion_awareness"),
    ("C3 GRPO think vs letter-only, heldout", "grpo_h", "grpo_h_rl", None, None),
    ("SFT vs base, heldout", "base_h", "sft_h", None, None),
    ("GRPO vs base, heldout", "base_h", "grpo_h", None, None),
]

# Family for Holm correction: SFT-vs-base per RoboVista domain (collapse claim).
DOMAINS = ["agriculture", "autonomous_driving", "domestic", "industrial_manufacturing",
           "open_datasets", "surgical_robotics"]


def load_summaries():
    """model_key -> {question_id: bool_correct} plus metadata maps."""
    by_key, meta = {}, {}
    for path in sorted(glob.glob(str(ROOT / "results" / "summary_*.json"))):
        with open(path) as f:
            d = json.load(f)
        rows = {}
        for r in d["results"]:
            rows[r["question_id"]] = bool(r.get("is_correct"))
            meta[r["question_id"]] = {
                "domain": r.get("domain"), "ability_type": r.get("ability_type"),
            }
        by_key[d["model"]] = rows  # later timestamp wins (sorted order)
    return by_key, meta


def bootstrap_ci(flags, n_boot, rng):
    n = len(flags)
    if n == 0:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += flags[rng.randrange(n)]
        stats.append(s / n)
    stats.sort()
    return (stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)])


def mcnemar_exact(b, c):
    """Two-sided exact McNemar: b = a-right/b-wrong, c = a-wrong/b-right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided binomial test at p=0.5
    total = 0.0
    for i in range(0, n + 1):
        p_i = math.comb(n, i) * 0.5 ** n
        if i <= k or i >= n - k:
            total += p_i
    return min(1.0, total)


def paired_counts(rows_a, rows_b, meta, subset_key=None, subset_value=None):
    common = set(rows_a) & set(rows_b)
    if subset_key:
        common = {q for q in common if meta.get(q, {}).get(subset_key) == subset_value}
    b = sum(1 for q in common if rows_a[q] and not rows_b[q])   # b better in A
    c = sum(1 for q in common if not rows_a[q] and rows_b[q])   # c better in B
    n = len(common)
    acc_a = sum(rows_a[q] for q in common) / n if n else float("nan")
    acc_b = sum(rows_b[q] for q in common) / n if n else float("nan")
    return n, acc_a, acc_b, b, c


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    parser = argparse.ArgumentParser(description="Stats report for RoboVista-R1")
    parser.add_argument("--boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(ROOT / "rl_runs"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_key, meta = load_summaries()
    report = {"runs": {}, "comparisons": [], "collapse_family": [], "missing": []}

    # ---- per-run accuracies with bootstrap CIs, aggregated over seeds
    for run, (label, seeds) in RUNS.items():
        entry = {"label": label, "seeds": {}, "mean": None, "sd": None}
        accs = []
        for seed, key in seeds.items():
            rows = by_key.get(key)
            if rows is None:
                report["missing"].append(key)
                continue
            flags = [int(v) for v in rows.values()]
            acc = sum(flags) / len(flags)
            lo, hi = bootstrap_ci(flags, args.boot, rng)
            entry["seeds"][seed] = {"model_key": key, "n": len(flags), "acc": acc,
                                    "ci95": [lo, hi]}
            accs.append(acc)
        if accs:
            m = sum(accs) / len(accs)
            entry["mean"] = m
            entry["sd"] = (sum((a - m) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5 if len(accs) > 1 else 0.0
        report["runs"][run] = entry

    # ---- pre-registered paired comparisons (per seed + pooled discordants)
    for name, ra, rb, sk, sv in COMPARISONS:
        seeds_a, seeds_b = RUNS[ra][1], RUNS[rb][1]
        per_seed, pooled_b, pooled_c = [], 0, 0
        for seed in sorted(set(seeds_a) & set(seeds_b) | {s for s in seeds_b if 0 in seeds_a}):
            key_a = seeds_a.get(seed, seeds_a.get(0))   # base runs exist only for seed 0
            key_b = seeds_b.get(seed)
            if key_b is None or key_a not in by_key or key_b not in by_key:
                continue
            n, acc_a, acc_b, b, c = paired_counts(by_key[key_a], by_key[key_b], meta, sk, sv)
            p = mcnemar_exact(b, c)
            per_seed.append({"seed": seed, "n": n, "acc_a": acc_a, "acc_b": acc_b,
                             "b_a_only": b, "c_b_only": c, "p_mcnemar": p})
            pooled_b += b
            pooled_c += c
        pooled_p = mcnemar_exact(pooled_b, pooled_c)
        report["comparisons"].append({
            "name": name, "a": ra, "b": rb, "subset": [sk, sv] if sk else None,
            "per_seed": per_seed,
            "pooled": {"b": pooled_b, "c": pooled_c, "p_mcnemar": pooled_p},
        })

    # ---- collapse family: SFT vs base per domain, Holm-corrected (pooled seeds)
    fam = []
    for dom in DOMAINS:
        pooled_b = pooled_c = 0
        accs_a, accs_b, n_dom = [], [], 0
        for seed, key_b in RUNS["sft"][1].items():
            key_a = RUNS["base"][1][0]
            if key_b not in by_key:
                continue
            n, acc_a, acc_b, b, c = paired_counts(by_key[key_a], by_key[key_b], meta, "domain", dom)
            pooled_b += b; pooled_c += c
            accs_a.append(acc_a); accs_b.append(acc_b); n_dom = n
        fam.append({"domain": dom, "n": n_dom,
                    "acc_base": accs_a[0] if accs_a else None,
                    "acc_sft_mean": sum(accs_b) / len(accs_b) if accs_b else None,
                    "b": pooled_b, "c": pooled_c,
                    "p_raw": mcnemar_exact(pooled_b, pooled_c)})
    for row, p_adj in zip(fam, holm([r["p_raw"] for r in fam])):
        row["p_holm"] = p_adj
    report["collapse_family"] = fam

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stats_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---- markdown rendering
    lines = ["# RoboVista-R1 statistics report", ""]
    lines += ["## Run accuracies (bootstrap 95% CI per seed; mean ± sd across seeds)", ""]
    lines += ["| Run | Seeds | Acc mean ± sd | Per-seed acc [95% CI] |", "|---|---|---|---|"]
    for run, e in report["runs"].items():
        if not e["seeds"]:
            continue
        per = "; ".join(f"s{s}: {v['acc']:.1%} [{v['ci95'][0]:.1%}, {v['ci95'][1]:.1%}]"
                        for s, v in sorted(e["seeds"].items()))
        sd = f" ± {e['sd']:.1%}" if len(e["seeds"]) > 1 else ""
        lines.append(f"| {e['label']} | {len(e['seeds'])} | {e['mean']:.1%}{sd} | {per} |")
    lines += ["", "## Pre-registered paired comparisons (exact McNemar)", ""]
    lines += ["| Comparison | n | acc A→B (seed 0) | discordants pooled (b/c) | p pooled | per-seed p |", "|---|---|---|---|---|---|"]
    for c in report["comparisons"]:
        if not c["per_seed"]:
            continue
        s0 = c["per_seed"][0]
        per = "; ".join(f"s{x['seed']}: {x['p_mcnemar']:.3g}" for x in c["per_seed"])
        lines.append(f"| {c['name']} | {s0['n']} | {s0['acc_a']:.1%} → {s0['acc_b']:.1%} | "
                     f"{c['pooled']['b']}/{c['pooled']['c']} | {c['pooled']['p_mcnemar']:.3g} | {per} |")
    lines += ["", "## SFT domain-collapse family (pooled seeds, Holm-corrected)", ""]
    lines += ["| Domain | n | base | SFT mean | b/c | p raw | p Holm |", "|---|---|---|---|---|---|---|"]
    for r in report["collapse_family"]:
        lines.append(f"| {r['domain']} | {r['n']} | {r['acc_base']:.1%} | {r['acc_sft_mean']:.1%} | "
                     f"{r['b']}/{r['c']} | {r['p_raw']:.3g} | {r['p_holm']:.3g} |")
    if report["missing"]:
        lines += ["", f"Missing summaries (not yet evaluated): {', '.join(sorted(set(report['missing'])))}"]
    with open(out / "stats_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
