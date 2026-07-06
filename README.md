# RoboVista

A crowdsourced visual question answering dataset for robotics, featuring multiple-choice questions with detailed reasoning.

Each question includes:

- One or more images from robotic manipulation scenes
- A question requiring visual understanding
- Up to five answer choices (A–E)
- The correct answer
- Detailed reasoning explaining the answer
- Category labels: domain, task, and ability type

The dataset is hosted on the HuggingFace Hub: [`sy-xie/robovista`](https://huggingface.co/datasets/sy-xie/robovista).

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

- `standard` — answer with the letter only
- `cot` — chain-of-thought reasoning before the final answer

Results are written to `results/summary_<model>_<timestamp>.json` with per-question responses and overall accuracy. Interrupted runs resume automatically from intermediate checkpoints.

## Dataset Fields

| Field | Type | Description |
|-------|------|-------------|
| `images` | list of images | One or more images for the question |
| `question` | string | The question text |
| `choices` | list of strings | Answer options A–E |
| `correct_answer` | string | The correct option letter |
| `reasoning` | string | Explanation for the answer |
| `domain` | string | Scene domain (e.g. kitchen, tabletop) |
| `task` | string | Robotic task category |
| `ability_type` | string | Ability being tested |
| `ability_subcategory` | string | Finer-grained ability label |
| `publication_source` | string | Source of the images |
| `id` | string | Unique question id |

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{robovista2026,
  title={RoboVista: A Crowdsourced Visual Question Answering Dataset for Robotics},
  year={2026},
  url={https://huggingface.co/datasets/sy-xie/robovista}
}
```
