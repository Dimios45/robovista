#!/usr/bin/env bash
# End-to-end reproduction of the RoboVista-R1 experiment (rl/BLOG.md).
#
# Prereqs:
#   - MODEL_PATH pointing at a local Qwen2-VL-7B-Instruct checkout
#   - a python env with rl/requirements.txt (torch matching your accelerator)
#   - a second env (or the same one) with `datasets` for the two export steps
#   - one ~80 GB GPU; ~12 GPU-hours total for all six training runs
#
# Usage:
#   MODEL_PATH=/path/to/Qwen2-VL-7B-Instruct ./rl/reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a Qwen2-VL-7B-Instruct directory}"
PY="${PY:-python}"            # env with rl/requirements.txt
DATA_PY="${DATA_PY:-$PY}"     # env that additionally has `datasets`
SEEDS="${SEEDS:-0 1 2}"

# 1. Data: RoboVista eval export + Robo2VLM-1 training subset (streamed, no 107 GB download)
[ -d data_local ]      || $DATA_PY benchmark/export_local.py --out data_local
[ -d rl_data/train ]   || $DATA_PY rl/export_robo2vlm.py --train 5000 --heldout 500 --out rl_data

# 2. Zero-shot baselines on RoboVista (letter-only, CoT, think-format, ICL)
$PY benchmark/run_benchmark_local.py --data-dir data_local --model-path "$MODEL_PATH" \
    --prompts standard cot rl --batch-size 8 --output-dir results
$PY benchmark/run_benchmark_local.py --data-dir data_local --model-path "$MODEL_PATH" \
    --prompts standard --icl-k 2 --batch-size 4 --output-dir results
$PY benchmark/run_benchmark_local.py --data-dir rl_data/heldout --model-path "$MODEL_PATH" \
    --model-key qwen2vl-7b-base-heldout --prompts standard --batch-size 8 --output-dir results

# 3. Train + evaluate both methods, all seeds
for S in $SEEDS; do
  if [ "$S" = "0" ]; then SUF=""; else SUF="-s$S"; fi
  GDIR="rl_runs/grpo${SUF/-/_}"; SDIR="rl_runs/sft${SUF/-/_}"

  $PY rl/grpo_train.py --data-dir rl_data/train --model-path "$MODEL_PATH" \
      --output-dir "$GDIR" --steps 300 --seed "$S"
  $PY benchmark/run_benchmark_local.py --data-dir data_local --model-path "$MODEL_PATH" \
      --adapter "$GDIR/adapter_latest" --model-key "qwen2vl-7b-grpo$SUF" \
      --prompts standard rl --batch-size 8 --output-dir results
  $PY benchmark/run_benchmark_local.py --data-dir rl_data/heldout --model-path "$MODEL_PATH" \
      --adapter "$GDIR/adapter_latest" --model-key "qwen2vl-7b-grpo$SUF-heldout" \
      --prompts standard rl --batch-size 8 --output-dir results

  $PY rl/sft_train.py --data-dir rl_data/train --model-path "$MODEL_PATH" \
      --output-dir "$SDIR" --epochs 2 --seed "$S"
  $PY benchmark/run_benchmark_local.py --data-dir data_local --model-path "$MODEL_PATH" \
      --adapter "$SDIR/adapter_final" --model-key "qwen2vl-7b-sft$SUF" \
      --prompts standard --batch-size 8 --output-dir results
  $PY benchmark/run_benchmark_local.py --data-dir rl_data/heldout --model-path "$MODEL_PATH" \
      --adapter "$SDIR/adapter_final" --model-key "qwen2vl-7b-sft$SUF-heldout" \
      --prompts standard --batch-size 8 --output-dir results
done

# 4. Calibration probe (seed-0 adapters, both eval sets)
for CFG in "base-robovista::data_local" \
           "sft-robovista:rl_runs/sft/adapter_final:data_local" \
           "grpo-robovista:rl_runs/grpo/adapter_latest:data_local" \
           "base-heldout::rl_data/heldout" \
           "sft-heldout:rl_runs/sft/adapter_final:rl_data/heldout" \
           "grpo-heldout:rl_runs/grpo/adapter_latest:rl_data/heldout"; do
  IFS=: read -r KEY ADAPTER DATA <<<"$CFG"
  $PY rl/calibration_eval.py --data-dir "$DATA" --model-path "$MODEL_PATH" \
      ${ADAPTER:+--adapter "$ADAPTER"} --model-key "$KEY" --max-batch-patches 8000
done

# 5. Statistics, error analysis (uses the committed human labels), figures
$PY rl/stats.py
$PY rl/error_analysis.py
$PY rl/make_figures.py

echo "Done. See rl_runs/stats_report.md, rl_runs/error_analysis.md, rl/figures/."
echo "Optional: re-label errors yourself with \`python rl/error_labeler.py\` (needs gradio)."
