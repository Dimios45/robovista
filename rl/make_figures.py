#!/usr/bin/env python3
"""Generate the figures for rl/RESULTS.md from training logs and eval summaries.

Usage:
    python rl/make_figures.py          # writes rl/figures/*.png
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# Reference palette (dataviz skill): categorical slots, chrome, diverging pair.
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"   # base / SFT / GRPO
DIV_POS, DIV_NEG = "#2a78d6", "#e34948"
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "figure.facecolor": PAGE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def style(ax, xgrid=False):
    ax.grid(axis="x" if xgrid else "y")
    ax.grid(axis="y" if xgrid else "x", visible=False)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)


def rolling(vals, w=15):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - w + 1)
        out.append(sum(vals[lo:i + 1]) / (i + 1 - lo))
    return out


def latest(pattern):
    paths = sorted(glob.glob(str(ROOT / "results" / pattern)))
    with open(paths[-1]) as f:
        return json.load(f)


def acc_by(results, key):
    groups = defaultdict(lambda: [0, 0])
    for r in results:
        g = r.get(key) or "unknown"
        groups[g][0] += 1
        groups[g][1] += bool(r.get("is_correct"))
    return {g: (c / n, n) for g, (n, c) in groups.items()}


# ---------------------------------------------------------------- fig 1: GRPO curves
rows = [json.loads(l) for l in open(ROOT / "rl_runs/grpo/train_log.jsonl")]
steps = [r["step"] for r in rows]
panels = [
    ("Rollout accuracy", "accuracy", "Fraction of sampled answers that are correct\n(training distribution, 32-96 rollouts/step)"),
    ("Mean reward", "reward_mean", "Correctness (+1.0) plus format bonus (+0.2)"),
    ("Format compliance", "format_rate", "Well-formed <think>...</think> 'Answer: X' outputs"),
    ("KL vs base policy", "kl", "How far the policy has drifted from frozen Qwen2-VL"),
]
fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), dpi=200)
fig.suptitle("GRPO training telemetry — 300 steps on Robo2VLM-1 (one MI300X, ~4.5 h)",
             fontsize=13, fontweight="bold", color=INK, y=0.99)
for ax, (title, key, sub) in zip(axes.flat, panels):
    vals = [r[key] for r in rows]
    ax.plot(steps, vals, color=BLUE, lw=0.8, alpha=0.28)
    ax.plot(steps, rolling(vals), color=BLUE, lw=2)
    ax.set_title(title, loc="left")
    ax.text(0, 1.005, "", transform=ax.transAxes)
    ax.set_xlabel(sub, fontsize=8, color=MUTED, labelpad=6)
    ax.set_xlim(0, 300)
    style(ax)
    if key in ("accuracy", "format_rate"):
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    if key == "kl":
        # One anomalous batch spikes to ~9 and would flatten the trend.
        ax.set_ylim(0, 0.65)
        ax.text(252, 0.50, "spike to 9.0\n(clipped) →", fontsize=7.5, color=MUTED,
                ha="right", va="center")
    end = rolling(vals)[-1]
    ax.annotate(f"{end:.0%}" if key != "kl" and key != "reward_mean" else f"{end:.2f}",
                (300, end), xytext=(4, 0), textcoords="offset points",
                color=INK, fontweight="bold", fontsize=10, va="center")
fig.tight_layout(rect=(0, 0, 0.97, 0.96))
fig.savefig(OUT / "fig1_grpo_training.png", facecolor=PAGE, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------- fig 2: eval results (bars)
robovista = [
    ("Base · letter-only", 33.5, BLUE), ("Base · CoT", 31.0, BLUE),
    ("Base · think-format", 31.6, BLUE), ("Base · ICL k=2", 25.3, BLUE),
    ("SFT · letter-only", 36.5, AQUA),
    ("GRPO · letter-only", 35.2, YELLOW), ("GRPO · think-format", 32.7, YELLOW),
]
heldout = [
    ("Base · letter-only", 33.2, BLUE),
    ("SFT · letter-only", 77.6, AQUA),
    ("GRPO · letter-only", 39.4, YELLOW), ("GRPO · think-format", 40.6, YELLOW),
]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=200,
                             gridspec_kw={"width_ratios": [7, 4.6]})
fig.suptitle("Where the learning went — held-out benchmark vs training distribution",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left")
for ax, data, title, xmax in [
    (a1, robovista, "RoboVista (474 Qs, never trained on)", 50),
    (a2, heldout, "Robo2VLM-1 held-out slice (500 Qs, in-distribution)", 88),
]:
    names = [d[0] for d in data][::-1]
    vals = [d[1] for d in data][::-1]
    colors = [d[2] for d in data][::-1]
    bars = ax.barh(names, vals, color=colors, height=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(v + xmax * 0.012, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.axvline(20, color=MUTED, lw=1.2, ls=(0, (3, 3)), zorder=2)
    ax.text(20.6, len(data) - 0.58, "random = 20%", color=MUTED, fontsize=8, va="bottom")
    ax.set_xlim(0, xmax)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.tick_params(axis="y", labelcolor=INK)
    style(ax, xgrid=True)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(OUT / "fig2_eval_results.png", facecolor=PAGE, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------- fig 3: per-ability deltas (diverging)
base_std = latest("summary_Qwen2-VL-7B-Instruct_2*.json")["results"]
base_cot = latest("summary_Qwen2-VL-7B-Instruct_cot_*.json")["results"]
grpo_rl = latest("summary_qwen2vl-7b-grpo_rl_*.json")["results"]

ABILITIES = [
    ("scene_understanding", "Scene understanding"),
    ("low_level_motion_awareness", "Low-level motion awareness"),
    ("high_level_decision_making", "High-level decision making"),
    ("recovery_replanning_robustness", "Recovery & robustness"),
]
base_ab = acc_by(base_std, "ability_type")


def deltas(results):
    ab = acc_by(results, "ability_type")
    return [(label, (ab[k][0] - base_ab[k][0]) * 100, base_ab[k][1])
            for k, label in ABILITIES]


fig, axes_ = plt.subplots(1, 2, figsize=(11, 3.6), dpi=200, sharey=True)
fig.suptitle("Thinking helps planning, hurts perception — accuracy change vs base letter-only, by ability",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left")
for ax, results, title in [
    (axes_[0], base_cot, "Zero-shot CoT (before RL)"),
    (axes_[1], grpo_rl, "GRPO think-format (after RL)"),
]:
    data = deltas(results)[::-1]
    names = [f"{d[0]}  (n={d[2]})" for d in data]
    vals = [d[1] for d in data]
    colors = [DIV_POS if v >= 0 else DIV_NEG for v in vals]
    bars = ax.barh(names, vals, color=colors, height=0.58, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(v + (0.7 if v >= 0 else -0.7), b.get_y() + b.get_height() / 2,
                f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right",
                color=INK, fontsize=9.5, fontweight="bold")
    ax.axvline(0, color=BASELINE, lw=1.4, zorder=2)
    ax.set_xlim(-31, 12)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}")
    ax.tick_params(axis="y", labelcolor=INK)
    style(ax, xgrid=True)
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.savefig(OUT / "fig3_ability_deltas.png", facecolor=PAGE, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------ fig 4: domain × model heatmap
sft_std = latest("summary_qwen2vl-7b-sft_2*.json")["results"]
grpo_std = latest("summary_qwen2vl-7b-grpo_2*.json")["results"]
DOMAINS = [
    ("surgical_robotics", "Surgical"), ("open_datasets", "Open datasets"),
    ("autonomous_driving", "Driving"), ("domestic", "Domestic"),
    ("agriculture", "Agriculture"), ("industrial_manufacturing", "Industrial"),
]
MODELS = [("Base", base_std), ("SFT", sft_std), ("GRPO", grpo_std), ("GRPO think", grpo_rl)]
grid = [[acc_by(res, "domain")[d][0] * 100 for _, res in MODELS] for d, _ in DOMAINS]
ns = {d: acc_by(base_std, "domain")[d][1] for d, _ in DOMAINS}

fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=200)
fig.suptitle("RoboVista accuracy by domain — all fine-tunes, letter-only + GRPO think",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left")
lo, hi = 15, 50
for i, row in enumerate(grid):
    for j, v in enumerate(row):
        t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
        color = SEQ[round(t * (len(SEQ) - 1))]
        ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, color=color))
        ax.text(j + 0.5, i + 0.5, f"{v:.0f}", ha="center", va="center", fontsize=10,
                fontweight="bold", color="#ffffff" if t > 0.55 else INK)
ax.set_xlim(0, len(MODELS)); ax.set_ylim(len(DOMAINS), 0)
ax.set_xticks([j + 0.5 for j in range(len(MODELS))], [m for m, _ in MODELS], color=INK)
ax.set_yticks([i + 0.5 for i in range(len(DOMAINS))],
              [f"{label}  (n={ns[d]})" for d, label in DOMAINS], color=INK)
ax.tick_params(length=0); ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("Accuracy (%), random = 20", fontsize=8.5, color=MUTED, labelpad=8)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "fig4_domain_heatmap.png", facecolor=PAGE, bbox_inches="tight")
plt.close(fig)

print("Wrote", *[p.name for p in sorted(OUT.glob("*.png"))])
