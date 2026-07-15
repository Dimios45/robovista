# Verifiable-Reward RL Generalizes Where SFT Memorizes: A Controlled Study of Fine-Tuning a 7B VLM for Robot VQA

*Draft mini-paper — RoboVista-R1 project, July 2026*

## Abstract

We fine-tune Qwen2-VL-7B-Instruct for robot visual question answering with two methods
under identical conditions — the same 5,000 template-generated Robo2VLM-1 questions, the
same LoRA budget (r=16, 0.48% of parameters) — differing only in objective: supervised
answer cross-entropy (SFT) versus GRPO with verifiable multiple-choice rewards (RL).
Across three seeds, evaluated on the expert-annotated RoboVista benchmark (never trained
on), we find a consistent dissociation. **SFT learns the training distribution
dramatically (+46.3 points in-distribution, 33.2→79.5±1.7%) but its apparent transfer
(+3.7, p=0.011) is entirely attributable to RoboVista's one distribution-adjacent
domain; excluding it, SFT is net-negative (31.8→30.3%, p=0.38)**, collapses toward
random on unseen domains (driving 40→22%, agriculture 32→26%), abandons the base
model's answer prior (χ²=1010), and becomes overconfident (ECE 0.18→0.34). **GRPO
learns far less in-distribution (+6.3) but transfers to fully-foreign domains
(31.8→33.7%, p=0.004), edits surgically (48 answer flips pooled across seeds, 3:1
favorable, vs SFT's 440), and leaves calibration intact (ECE 0.20).** A human-labeled
taxonomy of 102 RL-model errors finds 56% remain perception-bound (misidentification
33%, spatial 23%), replicating RoboVista's central "perception is the bottleneck"
claim for RL-tuned models: reasoning trained by RL helps planning directionally but
cannot repair what the vision tower gets wrong.

## 1. Setup

**Models.** Base: Qwen2-VL-7B-Instruct. Both fine-tunes: LoRA r=16, α=32 on all LM
attention+MLP projections (40.4M params); vision tower frozen; bf16; one MI300X.

**Data.** Train: 5,000-question shuffled subset of Robo2VLM-1 (684k template-generated
MCQs from DROID/OXE trajectories; ground truth from robot sensor state). In-distribution
eval: 500 held-out questions. Out-of-distribution eval: RoboVista (474 expert-annotated
questions, 6 domains). RoboVista's `open_datasets` domain (n=144) is built with the
Robo2VLM framework and is treated as distribution-adjacent; the other five domains
(n=330) are fully foreign.

**GRPO.** 300 steps × 4 prompts × 8 sampled completions (T=1.0, top-p 0.95, top-k
disabled — see §6). Verifiable reward: +1.0 correct letter, +0.2 well-formed
`<think>…</think> Answer: X`. Group-normalized advantages, no value model; token-level
policy gradient + k3 KL (β=0.04) against the adapter-disabled reference; degenerate
(zero-variance) groups resampled. ~4.5 h/seed.

**SFT.** Cross-entropy on the answer letter only, 2 epochs, ~45 min/seed.

**Statistics.** 3 seeds per method. Bootstrap 95% CIs (10k resamples). Paired exact
McNemar tests, per seed and pooled over seeds (discordant pairs summed); Holm
correction across the 6-domain family. Unanswered = incorrect (matches RoboVista's
protocol). Pre-registered comparisons; the "excluding open_datasets" split is a
post-hoc secondary analysis and labeled as such.

## 2. Results

### 2.1 Headline accuracies (mean ± sd over 3 seeds)

| Model / prompt | RoboVista (474, OOD) | Robo2VLM held-out (500, in-dist) |
|---|---|---|
| Base, letter-only | 33.5% [29.3, 37.8] | 33.2% [29.2, 37.4] |
| Base, CoT | 31.0% | — |
| Base, think-format | 31.6% | — |
| Base, ICL k=2 | 25.3% | — |
| SFT | 37.3 ± 1.0% | 79.5 ± 1.7% |
| GRPO | 35.2 ± 0.2% | 39.5 ± 0.9% |
| GRPO, think-format | 33.1 ± 1.1% | 41.3 ± 1.7% |

Reference: random 20%; leaderboard Qwen2.5-VL-72B 44.3%, best (Gemini 2.5 Pro) 56.5%.
No prior Qwen2-VL-7B entry exists.

### 2.2 Transfer vs memorization (the central claim)

| Comparison (pooled 3 seeds, exact McNemar) | discordants b/c | p |
|---|---|---|
| SFT vs base, RoboVista overall | 193/247 | 0.011 |
| **SFT vs base, excluding open_datasets (n=330)** | 137/122 | **0.38 (net negative)** |
| GRPO vs base, RoboVista overall | 12/36 | **7.2×10⁻⁴** |
| **GRPO vs base, excluding open_datasets** | 11/30 | **0.0043** |
| SFT vs base, held-out | 83/777 | 4.5×10⁻¹⁴² |
| GRPO vs base, held-out | 12/106 | 5.8×10⁻²⁰ |

Per-domain (Holm-corrected), the **only** significant SFT change is open_datasets
(37.5→53.5%, p=1.9×10⁻⁶). The collapse-direction domains (driving 40→22%, agriculture
32→26%, industrial 26→23%) are individually non-significant at these n, but their
combined effect is what nulls SFT's foreign-domain transfer above. GRPO's per-domain
profile never moves more than ±4 points.

**Answer churn** makes the mechanism visible: pooled across seeds, SFT flips 440
answers (193 right→wrong); GRPO flips 48 (12 right→wrong; 3:1 favorable; letter-prior
χ² vs base: SFT 1010, GRPO 1.4). The KL leash plus on-policy sampling constrains GRPO
to local, mostly-beneficial edits; SFT's unconstrained distribution-matching overwrites
the base policy wholesale, including where its training templates provide no support.

### 2.3 Calibration

A–E restricted-softmax confidence from a single forward pass (letter argmax accuracy
tracks generative accuracy within ~2 points):

| Model | RoboVista ECE₁₅ (mean conf) | Held-out ECE₁₅ |
|---|---|---|
| Base | 0.184 (52%) | 0.141 |
| SFT | **0.338 (69%)** | 0.087 |
| GRPO | 0.197 (53%) | 0.121 |

SFT nearly doubles OOD calibration error — confidence +17 points for +2 accuracy —
while remaining well-calibrated in-distribution. GRPO is calibration-neutral. This
extends RoboVista's ICL-calibration finding to supervised fine-tuning, and shows RL
avoids the pathology.

### 2.4 Reasoning: the CoT trade-off after RL

Zero-shot, think-format prompting is a pure tax at 7B (overall −1.9 to −2.5; low-level
motion awareness −11.8). After GRPO, the think-format shows the frontier-style pattern
directionally — high-level decision making +6.7 at seed 0, perception still negative —
but **the ability-level effects do not reach significance** (pooled McNemar: planning
28/34, p=0.53; perception degradation 41/28, p=0.15), and the planning gain is
seed-variable. The in-distribution sign flip (think > letter-only for GRPO, reversed
for base) is directionally consistent across all seeds (41.3±1.7 vs 39.5±0.9) but
p=0.25 pooled. We report these as suggestive; n≈70 per ability class is underpowered
for 5–8 point effects (post-hoc power ≈ 0.3).

### 2.5 What still fails: human-labeled error taxonomy

102 GRPO-think RoboVista errors, stratified by ability and reweighted to the true error
population (labeler: this project's human author; single annotator):

| Primary failure | Share |
|---|---|
| Task reasoning (right scene, wrong plan/logic) | 40.9% |
| Misidentification (saw the wrong thing) | 33.2% |
| Spatial reasoning (right objects, wrong geometry) | 23.2% |
| Format / refusal / other | 2.7% |

Perception-bound failures (misidentification + spatial) remain **56%** of all errors
after RL — and the misidentification share (33.2%) closely matches RoboVista's reported
30.2% for small models under their own taxonomy. Wrong answers also carry *longer*
think-traces than right ones in every ability class (e.g. scene understanding median
318 vs 278 chars): the model reasons hardest where reasoning cannot help.

## 3. Related work (brief)

RoboVista (RSS 2026) introduced the benchmark and the perception-bottleneck /
CoT-trade-off findings at frontier scale; Robo2VLM (NeurIPS 2025 D&B) provides the
sensor-grounded VQA generation framework we train on. Our method is the
R1-style verifiable-reward recipe (GRPO without a value model) applied to
multiple-choice robot VQA, in the lineage of RLVR / R1-V / Visual-RFT-style visual
GRPO work. Our contribution is not a new method but a controlled SFT-vs-RL comparison
with a clean OOD benchmark and seed-replicated statistics at 7B.

## 4. Limitations

Three seeds per method; ability-level and domain-level claims are underpowered and
reported with exact p-values rather than asserted. One base model, one adapter budget,
one training-data source; the excluding-open-datasets split is post-hoc. The error
taxonomy has a single human annotator. GRPO saw only ~1.2k unique questions and its
reward curve had not plateaued; the transfer gap between GRPO and SFT could change
with scale.

## 5. Takeaways

1. **Objective, not data, determined generalization**: same questions, same adapter —
   SFT memorized the template distribution; verifiable-reward RL learned a small
   amount of transferable skill.
2. **RL's KL-anchored, on-policy updates edit surgically** (48 flips vs 440) and
   preserve calibration; SFT's distribution-matching overwrites indiscriminately and
   inflates confidence OOD.
3. **Perception remains the bottleneck after RL** (56% of errors), confirming
   RoboVista's thesis for RL-tuned models — reward on answers cannot fix the vision
   tower it never trains. Perception-side interventions (e.g. grounding-tool-augmented
   inputs) are the natural next lever.

## 6. Reproducibility

Code: `rl/` (GRPO/SFT trainers, data exporter, stats engine, calibration eval, error
labeler, figure generator); evals via `benchmark/run_benchmark_local.py --adapter`.
Stats: `rl_runs/stats_report.{json,md}`; taxonomy: `rl_runs/error_taxonomy.json`.
Figures: `rl/figures/fig1–fig6`. Five implementation pitfalls that silently corrupt
results (greedy `top_k=1` default in Qwen2-VL's generation config; sdpa padding
garbage on ROCm; quadratic vision attention over batched images; left-padding label
misalignment; truncated think-format evals) are documented in `rl/RESULTS.md` §6 with
fixes in code.
