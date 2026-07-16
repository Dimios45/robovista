# RoboVista: Evaluating Vision-Language Models for Diverse Robot Applications

**Accepted to RSS 2026**

[**Project Page**](https://berkeleyautomation.github.io/robovista/) | [**Dataset (HuggingFace)**](https://huggingface.co/datasets/sy-xie/robovista) | [**Leaderboard**](https://berkeleyautomation.github.io/robovista/leaderboard.html) | Paper (coming soon)

Shuangyu Xie\*, Kaiyuan Chen\*, Ziyang Chen, Simeon Adebola, Yixuan Huang, Zehan Ma, Tianshuang Qiu, Wentao Yuan, Dhruv Shah, Pannag R. Sanketi, Ken Goldberg

*UC Berkeley · Princeton University · Google DeepMind* (\*equal contribution)

![RoboVista Overview](https://berkeleyautomation.github.io/robovista/static/images/teaser-rqa.png)

**TL;DR** — RoboVista is an expert-annotated, robot-centric Visual Question Answering benchmark built with **Robot Question Answering (RQA)**, a modular framework that turns real decision points from robot systems into grounded VQA. It contains **474 multiple-choice questions across 6 robot application domains and 39 task types**, from agriculture and surgery to industrial automation and autonomous driving. State-of-the-art VLMs show substantial gaps (best model only 56.5%), and physical robot experiments confirm that RoboVista performance strongly correlates with real-world task execution.

## The Benchmark

Each question pairs one or more robot-centric images with a five-option multiple choice question, the correct answer, and a detailed expert reasoning explanation. Visual data is curated from 18 peer-reviewed publications and open robot datasets (DROID, Open X-Embodiment, AgiBot). All annotators are graduate-level and above, with more than half holding Ph.D. degrees in Robotics.

| | |
|---|---|
| Expert-annotated VQA questions | **474** |
| Robot application domains | **6** |
| Distinct robot task types | **39** |
| Robot-centric images | **730** |

**Domains:**

| Domain | Questions | Description |
|--------|-----------|-------------|
| Open Datasets | 150 | DROID, Open X-Embodiment, and AgiBot trajectories, with questions based on the Robo2VLM framework |
| Industrial | 144 | 1D/3D deformable manipulation, assembly, bin picking, and defect scanning on factory lines |
| Agriculture | 62 | Robot gardening, plant inspection, and weed removal under occlusion and extreme lighting |
| Domestic | 52 | Home tidying and garment manipulation in well-lit, human-scale, structured scenes |
| Surgical | 46 | Long-horizon knot tying and debridement on the da Vinci Research Kit (dVRK) |
| Driving | 20 | Self-driving decision points from the autonomous-driving challenge in Mcity |

Questions span four functional layers of modular robot systems: **perception** (scene understanding), **high-level planning**, **action awareness** (motion feasibility), and **robustness** (failure detection and recovery).

## Results

Zero-shot accuracy (%) on RoboVista. Random baseline is 20%. See the [leaderboard](https://berkeleyautomation.github.io/robovista/leaderboard.html) for the full table.

| Model | All | Agri. | Driving | Home | Industry | Surgery | Open |
|-------|-----|-------|---------|------|----------|---------|------|
| **Gemini 2.5 Pro** | **56.5** | 48.4 | 50.0 | **63.2** | **48.4** | **76.1** | **58.3** |
| Qwen3-235B-A22B | 51.3 | 46.8 | **60.0** | 53.9 | 37.3 | 69.6 | 56.9 |
| GPT-4o | 49.6 | 50.0 | 50.0 | 59.2 | 32.5 | 67.4 | 53.5 |
| Qwen3-VL-32B | 49.2 | 48.4 | 55.0 | 48.7 | 35.7 | 65.2 | 55.6 |
| GPT-5 | 48.1 | 38.7 | 55.0 | 46.1 | 35.7 | 63.0 | 58.3 |
| RoboBrain 2.5-8B | 45.8 | 37.1 | 55.0 | 51.3 | 36.5 | 56.5 | 50.0 |
| Qwen2.5-VL-72B | 44.3 | 43.5 | 35.0 | 40.8 | 31.7 | 69.6 | 50.7 |
| Robo2VLM-ER | 42.6 | 35.5 | 40.0 | 43.4 | 32.5 | 54.3 | 50.7 |
| Qwen3-8B (text-only) | 25.1 | 27.4 | 30.0 | 30.8 | 22.2 | 26.1 | 24.0 |

**Key findings:**

- **RoboVista is hard** — even the best model reaches only 56.5% overall, and every domain leaves a substantial gap.
- **Domains differ sharply** — domestic scenes are easiest; agriculture (fine-grained plant morphology, self-occlusion, deformable structures) is consistently hardest.
- **Chain-of-Thought is a trade-off** — it degrades low-level perception by up to 12% through over-thinking, yet often improves multi-step planning.
- **In-context learning backfires** — same-domain examples reduce accuracy by 2.8–6.5% and raise calibration error by up to 9.7%.
- **Perception is the bottleneck** — misidentification is the dominant failure mode (30.2%); scale reduces it but does not resolve spatial reasoning.
- **The benchmark is predictive** — RoboVista scores strongly correlate with real-world bimanual alignment (ρ = −0.93) and surgical knot-tying progress on physical robots.

## Quick Start

### Load the dataset

```python
from datasets import load_dataset

ds = load_dataset("sy-xie/robovista", split="train")
print(ds[0]["question"])
print(ds[0]["choices"])
print(ds[0]["correct_answer"])
ds[0]["images"][0].show()  # PIL image
```

### Browse the dataset locally

An interactive gallery viewer with filtering by domain, task, and ability type:

```bash
pip install -r requirements.txt
python viewer/app.py
```

Then open http://localhost:7860. Click any tile to see the full question, images, answer choices, and reasoning.

You can also point the viewer at a local parquet export with `--parquet path/to/file.parquet`.

### Benchmark a model

Evaluate any vision-language model served through an OpenAI-compatible API (OpenAI, vLLM, SGLang, ...):

```bash
# OpenAI API
python benchmark/run_benchmark.py \
    --endpoint https://api.openai.com/v1 \
    --api-key $OPENAI_API_KEY \
    --model-id gpt-4o \
    --prompts standard cot

# Local vLLM / SGLang server
python benchmark/run_benchmark.py \
    --endpoint http://localhost:8000/v1 \
    --api-key sk-local \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --model-key qwen2.5-vl-7b
```

Two prompt configurations are included in `benchmark/prompts.json`:

- `standard` — answer with the letter only (matches the zero-shot numbers above)
- `cot` — chain-of-thought reasoning before the final answer

Useful flags: `--max-questions 10` for a quick smoke test, `--concurrency 10` for faster runs. Results are written to `results/summary_<model>_<timestamp>.json` with per-question responses and overall accuracy. Interrupted runs resume automatically from intermediate checkpoints.

## Dataset Fields

| Field | Type | Description |
|-------|------|-------------|
| `images` | list of images | One or more robot-centric images for the question |
| `question` | string | The question text |
| `choices` | list of strings | Answer options A–E |
| `correct_answer` | string | The correct option letter |
| `reasoning` | string | Expert explanation for the answer |
| `domain` | string | Application domain (agriculture, driving, domestic, industrial, surgical, open datasets) |
| `task` | string | Robot task category (39 types) |
| `ability_type` | string | Functional layer being tested (perception, planning, action awareness, robustness) |
| `ability_subcategory` | string | Finer-grained ability label |
| `publication_source` | string | Source publication or dataset for the images |
| `id` | string | Unique question id |

## RoboVista-R1: RL fine-tuning experiment

This repo also contains a full GRPO-vs-SFT fine-tuning study of Qwen2-VL-7B on robot
VQA (3 seeds each, trained on a Robo2VLM-1 subset, evaluated on RoboVista): verifiable-
reward RL transfers to unseen robot domains where SFT memorizes its templates and
collapses. Read the story with all figures in **[rl/BLOG.md](rl/BLOG.md)**; reproduce
everything with `MODEL_PATH=<Qwen2-VL-7B-Instruct> ./rl/reproduce.sh` (deps:
`rl/requirements.txt`; ~12 GPU-hours on one 80 GB+ GPU).

## Citation

```bibtex
@inproceedings{xie2026robovista,
  title     = {RoboVista: Evaluating Vision-Language Models for Diverse Robot Applications},
  author    = {Xie, Shuangyu and Chen, Kaiyuan and Chen, Ziyang and Adebola, Simeon and Huang, Yixuan and Ma, Zehan and Qiu, Tianshuang and Yuan, Wentao and Shah, Dhruv and Sanketi, Pannag R. and Goldberg, Ken},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2026},
}
```
