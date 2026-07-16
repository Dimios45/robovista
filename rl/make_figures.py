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
# Accuracies come from rl_runs/stats_report.json (run `python rl/stats.py` first):
# seed-0 for base variants, mean across seeds for the fine-tunes.
STATS_PATH = ROOT / "rl_runs" / "stats_report.json"
if not STATS_PATH.exists():
    raise SystemExit("rl_runs/stats_report.json missing — run `python rl/stats.py` first")
STATS = json.load(open(STATS_PATH))


def run_acc(run):
    e = STATS["runs"][run]
    return 100 * (e["mean"] if len(e["seeds"]) > 1 else e["seeds"]["0"]["acc"])


robovista = [
    ("Base · letter-only", run_acc("base"), BLUE), ("Base · CoT", run_acc("base_cot"), BLUE),
    ("Base · think-format", run_acc("base_rl"), BLUE), ("Base · ICL k=2", run_acc("base_icl"), BLUE),
    ("SFT · letter-only", run_acc("sft"), AQUA),
    ("GRPO · letter-only", run_acc("grpo"), YELLOW), ("GRPO · think-format", run_acc("grpo_rl"), YELLOW),
]
heldout = [
    ("Base · letter-only", run_acc("base_h"), BLUE),
    ("SFT · letter-only", run_acc("sft_h"), AQUA),
    ("GRPO · letter-only", run_acc("grpo_h"), YELLOW), ("GRPO · think-format", run_acc("grpo_h_rl"), YELLOW),
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

# ------------------------------------------------- fig 5: reliability diagram
import os
if all(os.path.exists(ROOT / "results" / f"calibration_{k}.json")
       for k in ["base-robovista", "sft-robovista", "grpo-robovista",
                 "base-heldout", "sft-heldout", "grpo-heldout"]):
    def rel_curve(path, bins=10):
        d = json.load(open(path))
        pts = []
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            sel = [r for r in d["records"] if lo < r["confidence"] <= hi]
            if len(sel) >= 8:
                pts.append((sum(r["confidence"] for r in sel) / len(sel),
                            sum(r["is_correct"] for r in sel) / len(sel)))
        return pts, d["ece15"]

    fig, axes_ = plt.subplots(1, 2, figsize=(10, 4.3), dpi=200)
    fig.suptitle("Calibration — SFT becomes overconfident out-of-distribution; GRPO stays calibrated",
                 fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left")
    for ax, suffix, title in [(axes_[0], "robovista", "RoboVista (out-of-distribution)"),
                              (axes_[1], "heldout", "Robo2VLM held-out (in-distribution)")]:
        ax.plot([0, 1], [0, 1], color=BASELINE, lw=1.2, ls=(0, (3, 3)), zorder=1)
        for name, key, color in [("Base", "base", BLUE), ("SFT", "sft", AQUA), ("GRPO", "grpo", YELLOW)]:
            pts, e = rel_curve(ROOT / "results" / f"calibration_{key}-{suffix}.json")
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=5, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax.annotate(f"{name} (ECE {e:.2f})", (xs[-1], ys[-1]), xytext=(6, 0),
                        textcoords="offset points", color=INK, fontsize=8.5,
                        fontweight="bold", va="center")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel("Model confidence (A–E softmax)", fontsize=8.5, color=MUTED)
        ax.set_ylabel("Actual accuracy", fontsize=8.5, color=MUTED)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        style(ax)
    fig.tight_layout(rect=(0, 0, 0.94, 0.90))
    fig.savefig(OUT / "fig5_reliability.png", facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------- fig 6: answer churn (flip counts)
def pooled_flips(comparison_name):
    for c in STATS["comparisons"]:
        if c["name"] == comparison_name:
            return c["pooled"]["b"], c["pooled"]["c"]
    raise KeyError(comparison_name)


sft_b, sft_c = pooled_flips("C1 SFT vs base, RoboVista overall")
grpo_b, grpo_c = pooled_flips("GRPO vs base, RoboVista overall")
CHURN = [
    ("SFT vs base", sft_b, sft_c, AQUA),
    ("GRPO vs base", grpo_b, grpo_c, YELLOW),
]
fig, ax = plt.subplots(figsize=(8.6, 2.9), dpi=200)
fig.suptitle("Answer churn on RoboVista (pooled over 3 seeds) — GRPO edits surgically",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left")
ys = range(len(CHURN))
for i, (name, unl, lrn, color) in enumerate(CHURN):
    ax.barh(i + 0.18, -unl, height=0.32, color=DIV_NEG, zorder=3)
    ax.barh(i - 0.18, lrn, height=0.32, color=DIV_POS, zorder=3)
    ax.text(-unl - 6, i + 0.18, f"{unl} unlearned", ha="right", va="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.text(lrn + 6, i - 0.18, f"{lrn} learned", ha="left", va="center", color=INK, fontsize=9.5, fontweight="bold")
ax.axvline(0, color=BASELINE, lw=1.4, zorder=2)
ax.set_yticks(list(ys), [c[0] for c in CHURN], color=INK)
ax.set_xlim(-320, 340)
ax.set_xlabel("questions flipped right→wrong (red)  |  wrong→right (blue), of 3×474", fontsize=8.5, color=MUTED)
ax.xaxis.set_major_formatter(lambda v, _: f"{abs(v):.0f}")
style(ax, xgrid=True)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(OUT / "fig6_answer_churn.png", facecolor=PAGE, bbox_inches="tight")
plt.close(fig)

print("Wrote fig5, fig6")
