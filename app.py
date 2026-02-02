#!/usr/bin/env python3
"""
Flask web app for easily converting lecture slides into Anki flashcards for study
"""

import os
import re
import csv
import time
import json
import uuid
from typing import List, Dict
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pypdf import PdfReader
from openai import OpenAI
import genanki

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MAX_PDF_PAGES = 100  # Maximum pages allowed
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Store progress so that progress bar is possible
progress_store = {}
progress_lock = threading.Lock()

CONCURRENT_SLIDES = 3  # Number of slides to process in parallel


def update_progress(job_id: str, **kwargs):
    """Thread-safe update of progress store."""
    with progress_lock:
        if job_id in progress_store:
            progress_store[job_id].update(kwargs)


def get_progress_snapshot(job_id: str):
    """Thread-safe read of progress store."""
    with progress_lock:
        if job_id in progress_store:
            return dict(progress_store[job_id])
        return None

SYSTEM_PROMPT = """You are an expert teaching assistant that writes concise, high-quality flashcards for spaced repetition.
Target audience: university students preparing for exams.

Rules:
- Prefer clear, atomic Q/A cards that test one concept at a time.
- Avoid trivialities (e.g., "What is slide 3's title?", "Which three cybersecurity frameworks are listed on slide 4?". Keep the questions and answers broad, covering the ideas rather than unimportant information).
- If the slide is mostly images or contains no study-worthy content, produce no cards.
- Keep answers short and factual; avoid fluff.
- If a formula or definition appears, make at least one card for it.
- Output MUST be NDJSON where each line is a JSON object with keys: front, back.
  Example line: {"front":"What is Fitts' Law?","back":"It models pointing time as a function of distance and target width."}
"""

USER_PROMPT_TEMPLATE = """Create up to {max_cards_per_slide} excellent flashcards from this lecture slide content.

Slide number: {slide_idx}
Slide text:
---
{slide_text}
---

Remember: Output NDJSON (one JSON object per line). Keys: front, back.
If no good cards, output nothing for this slide.
"""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_text(t: str) -> str:
    t = t.replace("\r", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    return t

def extract_pdf_text_by_page(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = normalize_text(text)
        pages.append(text)
    return pages

def chunk_slide_text(t: str, max_chars: int = 5000) -> List[str]:
    if len(t) <= max_chars:
        return [t]
    chunks = []
    start = 0
    while start < len(t):
        end = start + max_chars
        chunks.append(t[start:end])
        start = end
    return chunks

def call_openai_for_cards(client, slide_text: str, slide_idx: int, max_per_slide: int, model: str) -> List[Dict[str, str]]:
    cards: List[Dict[str, str]] = []
    chunks = chunk_slide_text(slide_text, max_chars=5000)

    for i, chunk in enumerate(chunks):
        user_prompt = USER_PROMPT_TEMPLATE.format(
            max_cards_per_slide=max_per_slide,
            slide_idx=f"{slide_idx}{'' if len(chunks)==1 else f' (part {i+1}/{len(chunks)})'}",
            slide_text=chunk
        )

        streamed_text = ""

        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            stream=True
        )

        for event in resp:
            if event.type == "response.output_text.delta":
                streamed_text += event.delta

        for line in streamed_text.splitlines():
            line = line.strip()
            if not line:
                continue

            # remove accidental ```json blocks which would mess up JSON parsing
            line = re.sub(r"^```(?:json)?", "", line).strip()
            line = re.sub(r"```$", "", line).strip()

            try:
                obj = json.loads(line)
                # Expect {"front": "...", "back": "..."}
                if "front" in obj and "back" in obj:
                    front = obj["front"].strip()
                    back = obj["back"].strip()
                    if front and back:
                        cards.append({"front": front, "back": back})
            except json.JSONDecodeError:
                # Ignore non-JSON lines
                continue

        time.sleep(0.3)

    return cards


def process_pdf(job_id: str, pdf_path: str, params: dict):
    """Process PDF in background thread with concurrent slide processing."""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        
        client = OpenAI()

        # Extract text from the pages
        pages = extract_pdf_text_by_page(pdf_path)
        total_pages = len(pages)
        
        if total_pages == 0:
            raise RuntimeError("No pages extracted from the provided PDF.")
        
        if total_pages > MAX_PDF_PAGES:
            raise RuntimeError(f"PDF has {total_pages} pages. Maximum allowed is {MAX_PDF_PAGES} pages.")

        update_progress(job_id, total_pages=total_pages)

        cards: List[Dict[str, str]] = []
        global_cap = params.get('max_cards', 300)
        max_per_slide = params.get('max_per_slide', 3)
        model = params.get('model', 'gpt-4o-mini')
        start_page = params.get('start_page', 1)
        end_page = params.get('end_page', total_pages)
        skip_empty = params.get('skip_empty', False)

        # Build list of slides to process
        slides_to_process = []
        for idx0, slide_text in enumerate(pages):
            slide_idx = idx0 + 1
            
            if slide_idx < start_page or slide_idx > end_page:
                continue
            if skip_empty and not slide_text:
                continue
            slides_to_process.append((slide_idx, slide_text))

        total_slides = len(slides_to_process)
        slides_completed = 0

        # Process slides in batches for concurrency while respecting global cap
        for batch_start in range(0, total_slides, CONCURRENT_SLIDES):
            if len(cards) >= global_cap:
                break

            batch = slides_to_process[batch_start:batch_start + CONCURRENT_SLIDES]

            with ThreadPoolExecutor(max_workers=CONCURRENT_SLIDES) as executor:
                futures = {}
                for slide_idx, slide_text in batch:
                    future = executor.submit(
                        call_openai_for_cards,
                        client=client,
                        slide_text=slide_text,
                        slide_idx=slide_idx,
                        max_per_slide=max_per_slide,
                        model=model,
                    )
                    futures[future] = slide_idx

                for future in as_completed(futures):
                    slide_idx = futures[future]
                    slides_completed += 1
                    try:
                        proposed = future.result()
                        if proposed:
                            space = global_cap - len(cards)
                            if space > 0:
                                cards.extend(proposed[:space])
                    except Exception as e:
                        print(f"[warn] Slide {slide_idx} failed: {e}")

                    update_progress(job_id,
                        current_page=slide_idx,
                        progress=int((slides_completed / total_slides) * 100),
                        cards_generated=len(cards),
                    )

        if not cards:
            raise RuntimeError("No cards generated. Check PDF text extraction and try different parameters.")

        # Write CSV
        csv_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Front", "Back", "Tags"])
            for c in cards:
                writer.writerow([c.get("front", ""), c.get("back", ""), c.get("tags", "")])

        # Create anki deck if user has specified to do so
        apkg_path = None
        if params.get('generate_apkg', True):
            deck_name = params.get('deck_name', 'Lecture Flashcards')
            apkg_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}.apkg")
            build_apkg_from_csv(csv_path, deck_name, apkg_path)

        with progress_lock:
            progress_store[job_id] = {
                'status': 'completed',
                'progress': 100,
                'total_pages': total_pages,
                'current_page': total_pages,
                'cards_generated': len(cards),
                'error': None,
                'csv_path': csv_path,
                'apkg_path': apkg_path,
            }

    except Exception as e:
        with progress_lock:
            progress_store[job_id] = {
                'status': 'error',
                'error': str(e),
                'progress': progress_store.get(job_id, {}).get('progress', 0),
            }


def build_apkg_from_csv(csv_path: str, deck_name: str, apkg_path: str):
    deck_id = abs(hash(deck_name)) % (10**10) # unique deck ID for each deck created
    deck = genanki.Deck(deck_id, deck_name)

    model = genanki.Model(
        1292005120, # unique model ID so that Anki can keep track of the model. Required by the genanki library
        'Simple Model',
        fields=[
            {'name': 'Front'},
            {'name': 'Back'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Front}}',
                'afmt': '{{Front}}<hr id="answer">{{Back}}',
            },
        ],
        css="""
        .card { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; font-size: 18px; }
        """
    )

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            front = row.get("Front", "").strip()
            back = row.get("Back", "").strip()
            if not front or not back: # should not happen, but just in case openai does some weird stuff
                continue
            note = genanki.Note(
                model=model,
                fields=[front, back],
            )
            deck.add_note(note)

    pkg = genanki.Package(deck)
    pkg.write_to_file(apkg_path)


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400

    params = request.form.to_dict()
    
    # Validate PDF page count first
    temp_path = None
    try:
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}.pdf")
        file.save(temp_path)
        
        reader = PdfReader(temp_path)
        page_count = len(reader.pages)
        
        if page_count > MAX_PDF_PAGES:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({
                'error': f'PDF has {page_count} pages. Maximum allowed is {MAX_PDF_PAGES} pages.'
            }), 400
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        final_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.pdf")
        os.rename(temp_path, final_path)
        temp_path = None  # File moved, don't delete it
        
        # Parse parameters
        end_page_param = params.get('end_page', '')
        end_page_value = int(end_page_param) if end_page_param else page_count
        
        params_dict = {
            'max_cards': int(params.get('max_cards', 300)),
            'max_per_slide': int(params.get('max_per_slide', 3)),
            'model': params.get('model', 'gpt-5-nano'),
            'start_page': int(params.get('start_page', 1)),
            'end_page': end_page_value,
            'skip_empty': params.get('skip_empty', 'false').lower() == 'true',
            'generate_apkg': params.get('generate_apkg', 'true').lower() == 'true',
            'deck_name': params.get('deck_name', 'Lecture Flashcards')
        }
        
        # Initialize progress BEFORE starting thread to prevent 404s on early polls
        with progress_lock:
            progress_store[job_id] = {
                'status': 'processing',
                'progress': 0,
                'total_pages': page_count,
                'current_page': 0,
                'cards_generated': 0,
                'error': None,
            }

        # Start processing in background
        thread = threading.Thread(target=process_pdf, args=(job_id, final_path, params_dict))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'page_count': page_count,
            'message': 'File uploaded and processing started'
        })
        
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500


@app.route('/api/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    snapshot = get_progress_snapshot(job_id)
    if snapshot is None:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(snapshot)

def _download_error(message: str, status_code: int):
    """Return error as JSON for fetch() requests, or HTML for direct browser navigation."""
    if request.headers.get('Accept', '').startswith('application/json'):
        return jsonify({'error': message}), status_code
    # HTML fallback for direct browser navigation (window.location.href)
    html = f"""<!DOCTYPE html>
<html><head><title>Download Error</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,sans-serif;display:flex;
justify-content:center;align-items:center;min-height:100vh;margin:0;background:#1a1a2e;color:#e0e0e0;}}
.box{{text-align:center;padding:2rem;border-radius:12px;background:#16213e;max-width:400px;}}
h2{{color:#e94560;}}a{{color:#0f3460;background:#e0e0e0;padding:8px 20px;border-radius:6px;
text-decoration:none;display:inline-block;margin-top:1rem;}}</style></head>
<body><div class="box"><h2>Download Error</h2><p>{message}</p>
<a href="/">Back to Converter</a></div></body></html>"""
    return Response(html, status=status_code, content_type='text/html')


@app.route('/api/download/<job_id>', methods=['GET'])
def download_file(job_id):
    file_type = request.args.get('type', 'csv')  # 'csv' or 'apkg'

    job = get_progress_snapshot(job_id)
    if job is None:
        return _download_error('Job not found. It may have expired — please generate your flashcards again.', 404)

    if job['status'] != 'completed':
        return _download_error('Job has not completed yet. Please wait for processing to finish.', 400)

    if file_type == 'csv':
        file_path = job.get('csv_path')
        if not file_path or not os.path.exists(file_path):
            return _download_error('CSV file not found. It may have been cleaned up - please generate again.', 404)
        return send_file(file_path, as_attachment=True, download_name=f'cards_{job_id}.csv')
    elif file_type == 'apkg':
        file_path = job.get('apkg_path')
        if not file_path or not os.path.exists(file_path):
            return _download_error('Anki deck file not found. It may have been cleaned up - please generate again.', 404)
        return send_file(file_path, as_attachment=True, download_name=f'cards_{job_id}.apkg')
    else:
        return _download_error('Invalid file type requested.', 400)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

