#!/usr/bin/env python3
"""Benchmark a vision-language model on the RoboVista dataset.

Works with any OpenAI-compatible chat completions endpoint (OpenAI, vLLM,
SGLang, etc.). Loads the dataset from the HuggingFace Hub (or a local parquet
export) and reports multiple-choice accuracy.

Usage:
    python benchmark/run_benchmark.py \
        --endpoint https://api.openai.com/v1 \
        --api-key $OPENAI_API_KEY \
        --model-id gpt-4o \
        --prompts standard cot

    # Local vLLM / SGLang server:
    python benchmark/run_benchmark.py \
        --endpoint http://localhost:8000/v1 --api-key sk-local \
        --model-id Qwen/Qwen2.5-VL-7B-Instruct --model-key qwen2.5-vl-7b
"""
import argparse
import asyncio
import base64
import json
import random
import re
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from tqdm import tqdm

DEFAULT_DATASET = "sy-xie/robovista"
CHOICE_LETTERS = ["A", "B", "C", "D", "E"]

MAX_RETRIES = 10
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 60.0  # seconds
MAX_IMAGE_SIZE = 720  # max width or height in pixels
SAVE_INTERVAL = 10  # save intermediate results every N questions


def load_questions(dataset_id: str, parquet: Optional[str]):
    from datasets import Dataset, load_dataset

    if parquet:
        ds = Dataset.from_parquet(parquet)
    else:
        ds = load_dataset(dataset_id, split="train")

    # Read metadata without decoding images; images are fetched lazily per
    # question during the run to keep memory bounded.
    meta = ds.remove_columns([c for c in ds.column_names if c == "images"])
    questions = []
    for i, row in enumerate(meta):
        choices = {letter: text for letter, text in zip(CHOICE_LETTERS, row["choices"]) if text}
        if not choices:
            continue
        questions.append({
            "question_id": row.get("id", str(i)),
            "question": row["question"],
            "choices": choices,
            "correct_answer": row["correct_answer"],
            "index": i,
        })
    return ds, questions


def resize_image(img, max_size: int = MAX_IMAGE_SIZE) -> bytes:
    """Resize a PIL image to fit within max_size pixels and return JPEG bytes."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    width, height = img.size
    if width > max_size or height > max_size:
        if width > height:
            new_width, new_height = max_size, int(height * max_size / width)
        else:
            new_width, new_height = int(width * max_size / height), max_size
        img = img.resize((new_width, new_height))

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def format_question_with_choices(question: str, choices: Dict[str, str], suffix: str) -> str:
    formatted_choices = "\n".join(f"{letter}. {text}" for letter, text in choices.items() if text)
    return f"{question}\n\nOptions:\n{formatted_choices}\n\n{suffix}"


def _extract_letter(response_upper: str, strict: bool = False) -> Optional[str]:
    """Extract an answer letter from uppercased text. With strict=True, only
    explicit "answer is X" statements count (no loose single-letter matches)."""
    valid_letters = set(CHOICE_LETTERS)

    answer_patterns = [
        r"\b(?:THE\s+)?ANSWER\s*(?:IS|:)\s*([A-E])\b",
        r"\b([A-E])\s*(?:IS\s+(?:THE\s+)?(?:CORRECT|RIGHT|BEST)\s+ANSWER)\b",
        r"^\s*([A-E])\s*$",
        r"^\s*([A-E])\.\s*$",
        r"^\s*([A-E])\s*[\.\)]",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, response_upper)
        if match:
            return match.group(1)

    if strict:
        # A committed final statement like "... the best option is D." at the
        # very end of the text (e.g. just before a closing </think>).
        match = re.search(r"\b(?:IS|:)\s*([A-E])\b[^A-E]{0,40}$", response_upper.rstrip())
        if match:
            return match.group(1)
        return None

    if response_upper and response_upper[0] in valid_letters:
        if len(response_upper) == 1 or not response_upper[1].isalpha():
            return response_upper[0]

    response_stripped = response_upper.rstrip(".")
    if response_stripped and response_stripped[-1] in valid_letters:
        if len(response_stripped) == 1 or not response_stripped[-2].isalpha():
            return response_stripped[-1]

    for letter in valid_letters:
        if re.search(rf"\b{letter}\b", response_upper):
            return letter

    return None


def parse_answer(response: str) -> Optional[str]:
    """Parse model response to extract the answer letter."""
    if not response or not isinstance(response, str):
        return None

    think_match = re.search(r"</think>\s*(.*)$", response, re.IGNORECASE | re.DOTALL)
    if think_match:
        tail = think_match.group(1).strip().upper()
        answer = _extract_letter(tail)
        if answer:
            return answer
        # Some models state the answer inside the think block and stop at
        # </think>; look for an explicit answer statement in the full text.
        return _extract_letter(response.strip().upper(), strict=True)

    return _extract_letter(response.strip().upper())


def extract_response_content(message: Dict[str, Any]) -> str:
    """Extract text from an API response message, handling list content and
    reasoning_content fallbacks used by some serving stacks."""
    def extract_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item.get("text", "").strip())
                elif isinstance(item, str):
                    parts.append(item.strip())
            return "\n".join(p for p in parts if p)
        return ""

    content = extract_text(message.get("content"))
    if not content:
        content = extract_text(message.get("reasoning_content"))
    return re.sub(r"\s+", " ", content).strip() if content else ""


async def call_vlm_api(
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: str,
    model_id: str,
    prompt: str,
    images: List[bytes],
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call the chat completions API with retry and exponential backoff."""
    image_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(b).decode()}"}}
        for b in images
    ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": image_content + [{"type": "text", "text": prompt}]},
    ]

    # Reasoning models use max_completion_tokens and reject temperature.
    uses_completion_tokens = any(x in model_id.lower() for x in ["gpt-5", "o1", "o3"])
    payload = {"model": model_id, "messages": messages}
    if uses_completion_tokens:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = temperature

    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.post(
                f"{endpoint}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    backoff = min(INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF)
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            backoff = max(backoff, float(retry_after))
                        except ValueError:
                            pass
                    await asyncio.sleep(backoff)
                    continue
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            return extract_response_content(message)
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(min(INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF))
                continue
            raise

    raise last_exception or RuntimeError("Max retries exceeded")


async def run_benchmark(
    ds,
    questions: List[Dict[str, Any]],
    endpoint: str,
    api_key: str,
    model_id: str,
    model_key: str,
    prompt_config: Dict[str, str],
    prompt_name: str,
    output_dir: Path,
    concurrency: int,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    output_model_name = model_key.replace("/", "-")
    if prompt_name != "standard":
        output_model_name = f"{output_model_name}_{prompt_name}"

    intermediate_path = output_dir / f"intermediate_{output_model_name}.json"
    existing: Dict[str, Dict[str, Any]] = {}
    if intermediate_path.exists():
        with open(intermediate_path) as f:
            existing = {r["question_id"]: r for r in json.load(f)}
        print(f"Resuming: {len(existing)} questions already completed.")

    remaining = [q for q in questions if q["question_id"] not in existing]
    all_results = dict(existing)
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def process(client: httpx.AsyncClient, q: Dict[str, Any], pbar: tqdm):
        async with semaphore:
            prompt = format_question_with_choices(q["question"], q["choices"], prompt_config["question_suffix"])
            try:
                images = [resize_image(img) for img in ds[q["index"]]["images"]]
                response = await call_vlm_api(
                    client, endpoint, api_key, model_id, prompt, images,
                    prompt_config["system_prompt"], max_tokens, temperature,
                )
                predicted = parse_answer(response)
                result = {
                    "question_id": q["question_id"],
                    "predicted_answer": predicted,
                    "model_response": response,
                    "correct_answer": q["correct_answer"],
                    "is_correct": predicted == q["correct_answer"],
                }
            except Exception as e:
                result = {"question_id": q["question_id"], "error": str(e), "is_correct": False}

            async with lock:
                all_results[q["question_id"]] = result
                if len(all_results) % SAVE_INTERVAL == 0:
                    with open(intermediate_path, "w") as f:
                        json.dump(list(all_results.values()), f, indent=2)
            pbar.update(1)

    if remaining:
        async with httpx.AsyncClient(timeout=300) as client:
            with tqdm(total=len(questions), initial=len(existing), desc=f"{model_key} [{prompt_name}]") as pbar:
                await asyncio.gather(*(process(client, q, pbar) for q in remaining))

    results = [all_results[q["question_id"]] for q in questions if q["question_id"] in all_results]
    correct = sum(1 for r in results if r.get("is_correct"))
    errors = sum(1 for r in results if "error" in r)
    accuracy = correct / len(results) if results else 0.0

    summary = {
        "model": output_model_name,
        "model_id": model_id,
        "prompt": prompt_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_questions": len(results),
        "correct": correct,
        "errors": errors,
        "accuracy": accuracy,
        "results": results,
    }

    summary_path = output_dir / f"summary_{output_model_name}_{summary['timestamp']}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    intermediate_path.unlink(missing_ok=True)

    print(f"\n{output_model_name}: {correct}/{len(results)} correct ({accuracy:.1%}), {errors} errors")
    print(f"Saved {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark a VLM on the RoboVista dataset")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HuggingFace dataset repo id")
    parser.add_argument("--parquet", help="Local parquet file (overrides --dataset)")
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible API base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model-id", required=True, help="Model id sent to the API")
    parser.add_argument("--model-key", help="Short name used in output filenames (defaults to --model-id)")
    parser.add_argument("--prompt-config", default=str(Path(__file__).parent / "prompts.json"))
    parser.add_argument("--prompts", nargs="+", default=["standard"], help="Prompt configurations to run")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=10240)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-questions", type=int, help="Limit number of questions (for quick tests)")
    args = parser.parse_args()

    with open(args.prompt_config) as f:
        prompt_configs = json.load(f)
    for name in args.prompts:
        if name not in prompt_configs:
            sys.exit(f"Error: prompt '{name}' not found in {args.prompt_config} (available: {list(prompt_configs)})")

    print(f"Loading dataset {'from ' + args.parquet if args.parquet else args.dataset} ...")
    ds, questions = load_questions(args.dataset, args.parquet)
    if args.max_questions:
        questions = questions[:args.max_questions]
    print(f"Loaded {len(questions)} questions.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_key = args.model_key or args.model_id.split("/")[-1]

    for prompt_name in args.prompts:
        asyncio.run(run_benchmark(
            ds=ds,
            questions=questions,
            endpoint=args.endpoint.rstrip("/"),
            api_key=args.api_key,
            model_id=args.model_id,
            model_key=model_key,
            prompt_config=prompt_configs[prompt_name],
            prompt_name=prompt_name,
            output_dir=output_dir,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ))


if __name__ == "__main__":
    main()
