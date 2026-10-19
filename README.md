# JobMatch

A chat assistant that helps you explore Swedish data-science and technology jobs. It searches real JobTech ads and answers in English with source citations. You can also upload a PDF CV to compare your experience with job requirements.

## Setup

Use Python 3.11 or newer:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env
```

For OpenAI, add your API key to `OPENAI_API_KEY` in `.env`. Keep this file private.

## Run the app

Build the search index once using the included job dataset:

```bash
python -m jobmatch.cli build
```

Then start the app:

```bash
./run.sh
```

Open **http://127.0.0.1:8501**. The first build downloads the embedding model. Restart the app after changing code or `.env` settings.

### Use local Qwen instead of OpenAI

Install [Ollama](https://ollama.com/download). In one terminal, start the local server:

```bash
./run.sh --ollama
```

In another terminal, download Qwen once and start the app:

```bash
ollama pull qwen2.5:7b
./run.sh
```

Select **Ollama (local)** and `qwen2.5:7b` in the app's Model settings.

## Using your CV

1. Upload a PDF or paste text under **Your CV · PDF or text**.
2. Review the extracted text and correct any mistakes.
3. Enable consent for your selected model.
4. Click **Find matching roles** or ask a question about your experience.

Scanned PDFs use local Tesseract OCR. On macOS, install it with `brew install tesseract`. English and Swedish language packs are supported; the app warns if a requested pack is missing. Uploads are limited to 10 MB and 12 pages.

PDFs are processed locally. With consent, the reviewed CV text is sent to the selected model. Check recommendations against the original ads, since listings and requirements may change.

## Project folders

- `app.py` — chat interface.
- `jobmatch/` — data download, search, model connections and PDF/OCR processing.
- `data/` — saved job ads.
- `tests/` — automated tests.

## Tests

```bash
python -m pytest -q
```

