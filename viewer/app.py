#!/usr/bin/env python3
"""RoboVista dataset viewer.

Browse the RoboVista VQA dataset (loaded from the HuggingFace Hub or a local
parquet file) in an interactive gallery with filters and per-question detail.

Usage:
    python viewer/app.py                          # loads the dataset from the HF Hub
    python viewer/app.py --dataset user/repo      # a different Hub dataset repo
    python viewer/app.py --parquet path/to.parquet  # a local parquet export
"""
import argparse
import base64
import html as html_module
import io
import random

import gradio as gr

DEFAULT_DATASET = "sy-xie/robovista"
QUESTIONS_PER_PAGE = 48
CHOICE_LETTERS = ["A", "B", "C", "D", "E"]

# Populated in main(); images stay lazily decoded inside the Arrow-backed dataset.
DS = None
QUESTIONS = []


def load_data(dataset_id: str, parquet: str):
    from datasets import Dataset, load_dataset

    if parquet:
        ds = Dataset.from_parquet(parquet)
    else:
        ds = load_dataset(dataset_id, split="train")

    # Metadata only (no image decoding) for filtering and tiles.
    meta = ds.remove_columns([c for c in ds.column_names if c == "images"])
    questions = []
    for i, row in enumerate(meta):
        questions.append({
            "index": i,
            "id": row.get("id", str(i)),
            "question": row.get("question", ""),
            "choices": row.get("choices", []),
            "correct_answer": row.get("correct_answer", ""),
            "reasoning": row.get("reasoning", ""),
            "domain": (row.get("domain") or "").strip(),
            "task": (row.get("task") or "").strip(),
            "ability_type": (row.get("ability_type") or "").strip(),
            "publication_source": row.get("publication_source", ""),
        })
    return ds, questions


def pil_to_base64(img, max_size=(400, 300)) -> str:
    try:
        img = img.copy()
        img.thumbnail(max_size)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{data}"
    except Exception:
        return ""


def get_images(index):
    return DS[index]["images"] or []


def get_unique_values(questions, field):
    return sorted({q[field] for q in questions if q[field]})


def filter_questions(questions, domain, task, ability_type):
    filtered = questions
    if domain:
        filtered = [q for q in filtered if q["domain"] == domain]
    if task:
        filtered = [q for q in filtered if q["task"] == task]
    if ability_type:
        filtered = [q for q in filtered if q["ability_type"] == ability_type]
    return filtered


def render_question_tile(q, tile_index):
    esc = html_module.escape
    images = get_images(q["index"])

    img_html = ""
    if images:
        b64_src = pil_to_base64(images[0])
        if b64_src:
            img_html = f'<img src="{b64_src}" style="width:100%; height:80px; object-fit:cover; border-radius:4px 4px 0 0;">'

    tile_id = f"tile_{tile_index}"

    correct = q["correct_answer"]
    choices_html = ""
    for letter, choice_text in zip(CHOICE_LETTERS, q["choices"]):
        if not choice_text:
            continue
        is_correct = letter == correct
        bg = "#d4edda" if is_correct else "#f5f5f5"
        marker = " ✓" if is_correct else ""
        choices_html += f'<div style="padding:6px 8px; background:{bg}; border-radius:4px; margin:4px 0;"><b>{letter}.</b> {esc(choice_text)}{marker}</div>'

    all_images = ""
    for img in images:
        b64 = pil_to_base64(img, max_size=(800, 600))
        if b64:
            all_images += f'<img src="{b64}" style="max-width:100%; max-height:300px; margin:4px; border-radius:4px; object-fit:contain;">'

    return f'''
    <div class="tile" onclick="document.getElementById('modal_{tile_id}').style.display='flex'"
         style="width:140px; background:white; border:1px solid #ddd; border-radius:6px; cursor:pointer; overflow:hidden; transition:transform 0.2s, box-shadow 0.2s;">
        {img_html}
        <div style="padding:6px; font-size:10px;">
            <div style="font-size:9px; color:#999; font-family:monospace; margin-bottom:3px; word-break:break-all;">{esc(q['id'])}</div>
            <div style="background:#e3f2fd; color:#1976d2; padding:1px 4px; border-radius:8px; display:inline-block;">{esc((q['domain'] or 'N/A')[:15])}</div>
            <div style="background:#fff3e0; color:#e65100; padding:1px 4px; border-radius:8px; display:inline-block; margin-top:3px;">{esc((q['task'] or 'N/A')[:15])}</div>
            <div style="background:#f3e5f5; color:#7b1fa2; padding:1px 4px; border-radius:8px; display:inline-block; margin-top:3px;">{esc((q['ability_type'] or 'N/A')[:20])}</div>
        </div>
    </div>
    <div id="modal_{tile_id}" onclick="if(event.target===this)this.style.display='none'"
         style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); z-index:1000; justify-content:center; align-items:center; padding:20px; box-sizing:border-box;">
        <div style="background:white; border-radius:8px; max-width:700px; max-height:90vh; overflow-y:auto; padding:16px; position:relative;">
            <button onclick="this.parentElement.parentElement.style.display='none'"
                    style="position:absolute; top:8px; right:8px; border:none; background:#eee; border-radius:50%; width:28px; height:28px; cursor:pointer; font-size:16px;">✕</button>
            <div style="font-size:12px; color:#888; font-family:monospace; margin-bottom:8px; word-break:break-all;">{esc(q['id'])}</div>
            <div style="display:flex; flex-wrap:wrap; gap:4px; margin-bottom:8px;">
                <span style="background:#e3f2fd; color:#1976d2; padding:2px 8px; border-radius:12px; font-size:11px;">{esc(q['domain'] or 'N/A')}</span>
                <span style="background:#fff3e0; color:#e65100; padding:2px 8px; border-radius:12px; font-size:11px;">{esc(q['task'] or 'N/A')}</span>
                <span style="background:#f3e5f5; color:#7b1fa2; padding:2px 8px; border-radius:12px; font-size:11px;">{esc(q['ability_type'] or 'N/A')}</span>
            </div>
            <div style="display:flex; flex-wrap:wrap; justify-content:center; margin-bottom:12px;">{all_images}</div>
            <div style="font-size:14px; margin-bottom:12px; line-height:1.4;"><b>Question:</b> {esc(q['question'])}</div>
            <div style="margin-bottom:12px;">{choices_html}</div>
            <div style="background:#e8f5e9; padding:10px; border-radius:6px; font-size:13px;"><b>Reasoning:</b> {esc(q['reasoning'])}</div>
            {f'<div style="margin-top:8px; font-size:11px; color:#888;">Source: {esc(q["publication_source"])}</div>' if q['publication_source'] else ''}
        </div>
    </div>'''


def get_gallery_html(questions, page):
    if not questions:
        return '<div style="text-align:center; padding:40px; color:#666;"><h3>No questions found</h3><p>Try adjusting your filters.</p></div>'

    total = len(questions)
    total_pages = (total + QUESTIONS_PER_PAGE - 1) // QUESTIONS_PER_PAGE
    page = max(0, min(page, total_pages - 1))

    start_idx = page * QUESTIONS_PER_PAGE
    end_idx = min(start_idx + QUESTIONS_PER_PAGE, total)
    page_questions = questions[start_idx:end_idx]

    html = f'''
    <style>
        .tile:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
    </style>
    <div style="text-align:center; margin-bottom:8px; color:#888; font-size:12px;">
        {start_idx + 1}-{end_idx} of {total} (Page {page + 1}/{total_pages})
    </div>
    <div style="display:flex; flex-wrap:wrap; gap:8px; justify-content:center; padding:8px;">
    '''
    for i, q in enumerate(page_questions):
        html += render_question_tile(q, start_idx + i)
    html += '</div>'
    return html


class GalleryState:
    def __init__(self):
        self.filtered_questions = QUESTIONS.copy()
        random.shuffle(self.filtered_questions)
        self.current_page = 0

    def apply_filters(self, domain, task, ability_type):
        self.filtered_questions = filter_questions(QUESTIONS, domain, task, ability_type)
        random.shuffle(self.filtered_questions)
        self.current_page = 0
        return self.get_html()

    def shuffle(self):
        random.shuffle(self.filtered_questions)
        self.current_page = 0
        return self.get_html()

    def next_page(self):
        total_pages = max(1, (len(self.filtered_questions) + QUESTIONS_PER_PAGE - 1) // QUESTIONS_PER_PAGE)
        self.current_page = min(self.current_page + 1, total_pages - 1)
        return self.get_html()

    def prev_page(self):
        self.current_page = max(0, self.current_page - 1)
        return self.get_html()

    def get_html(self):
        return get_gallery_html(self.filtered_questions, self.current_page)


def build_app():
    gallery_state = GalleryState()
    all_domains = get_unique_values(QUESTIONS, "domain")
    all_tasks = get_unique_values(QUESTIONS, "task")
    all_ability_types = get_unique_values(QUESTIONS, "ability_type")

    with gr.Blocks(title="RoboVista Dataset Viewer") as demo:
        gr.Markdown("## RoboVista Dataset Viewer")

        with gr.Row():
            prev_btn_top = gr.Button("← Prev", size="sm")
            with gr.Column(scale=2):
                gr.Markdown(f"<center style='font-size:11px;color:#888;'>{len(QUESTIONS)} questions</center>")
            next_btn_top = gr.Button("Next →", size="sm")

        gallery_html = gr.HTML(value=gallery_state.get_html())

        with gr.Row():
            prev_btn_bottom = gr.Button("← Prev", size="sm", scale=1)
            next_btn_bottom = gr.Button("Next →", size="sm", scale=1)

        with gr.Row():
            domain_filter = gr.Dropdown(choices=[""] + all_domains, value="", label="Domain", scale=2)
            task_filter = gr.Dropdown(choices=[""] + all_tasks, value="", label="Task", scale=2)
            ability_type_filter = gr.Dropdown(choices=[""] + all_ability_types, value="", label="Ability Type", scale=2)
            apply_btn = gr.Button("Filter", variant="primary", scale=1)
            shuffle_btn = gr.Button("Shuffle", variant="secondary", scale=1)

        apply_btn.click(gallery_state.apply_filters, inputs=[domain_filter, task_filter, ability_type_filter], outputs=[gallery_html])
        shuffle_btn.click(gallery_state.shuffle, outputs=[gallery_html])
        prev_btn_top.click(gallery_state.prev_page, outputs=[gallery_html])
        next_btn_top.click(gallery_state.next_page, outputs=[gallery_html])
        prev_btn_bottom.click(gallery_state.prev_page, outputs=[gallery_html])
        next_btn_bottom.click(gallery_state.next_page, outputs=[gallery_html])

    return demo


def main():
    global DS, QUESTIONS
    parser = argparse.ArgumentParser(description="RoboVista dataset viewer")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HuggingFace dataset repo id")
    parser.add_argument("--parquet", help="Local parquet file (overrides --dataset)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    print(f"Loading dataset {'from ' + args.parquet if args.parquet else args.dataset} ...")
    DS, QUESTIONS = load_data(args.dataset, args.parquet)
    print(f"Loaded {len(QUESTIONS)} questions.")

    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
