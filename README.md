# JobMatch

A chat assistant for Swedish data-science and technology job ads. It retrieves Swedish/English source excerpts and answers in English with citations. Supports OpenAI, local Qwen through Ollama, PDF CV uploads, local OCR, and CV evidence comparison.

## Setup and run

Use Python 3.11+:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env
```

Set `OPENAI_API_KEY` in your local `.env` for OpenAI. Never submit that file. The model defaults to `gpt-4.1-mini` and can be changed with `OPENAI_MODEL`.

The included `data/data_science_jobs.jsonl` is the fixed experimental snapshot. Build the local index once on a new machine, then start the UI:

```bash
python -m jobmatch.cli build
./run.sh
```

Open http://127.0.0.1:8501. The first index build downloads the multilingual embedding model. This machine already has the working index and model cache. Restart the app after code or `.env` changes.

For local Qwen, install [Ollama](https://ollama.com/download), then use two terminals:

```bash
# Terminal 1: keep this running
./run.sh --ollama
# Terminal 2: download once, then start JobMatch
ollama pull qwen2.5:7b
./run.sh
```

Choose **Ollama (local)** and `qwen2.5:7b` in Model settings. The launcher uses `.cache/ollama` for model files; start Ollama through this launcher to reuse that cache.

## Using the app

Ask about skills, qualifications and roles, or choose location/occupation filters. Expand **Sources** to inspect the original excerpts and links. **Job library** shows the snapshot's matching ads; these counts are not market-wide statistics.

For CV matching, upload a PDF or paste text in **Your CV · PDF or text**, review/edit the text, enable consent, and click **Find matching roles** or ask about your experience. Provider changes and CV edits reset consent. **New chat** clears messages; **Clear conversation and CV** also removes the CV and upload.

PDF extraction and OCR run locally in memory. Scanned pages use Tesseract; **Force OCR** handles broken text layers. Install Tesseract (`brew install tesseract` on macOS) and English/Swedish language packs on a new machine. This machine already has both packs in `.cache/tessdata/`. Optional settings: `TESSERACT_PATH`, `OCR_TESSDATA`, `OCR_LANGUAGE` (default `eng+swe`). If a requested pack is unavailable, the app warns and uses an available pack.

Uploads are limited to 10 MB / 12 pages; extracted CV text is limited to 16,000 characters with a warning when shortened. Check OCR errors and remove identifying details before sharing. Uploaded PDFs are not sent to the model, saved to disk, or indexed. With consent, reviewed CV text and relevant job excerpts are sent to the selected provider. OpenAI uses `store=False`; provider retention policies still apply.

## Structure

```text
app.py                  Streamlit interface
run.sh                  Start the app, or Ollama with --ollama
requirements.txt        Pinned project dependencies
.env.example            Configuration template
.streamlit/config.toml  UI/server settings
jobmatch/               Ingestion, retrieval, generation, PDF/OCR and CLI
 data.py                Cleaning and tokenizer-based chunking
 ingest.py              JobTech snapshot download and scope filtering
 retrieval.py           E5 embeddings, Chroma, BM25L and rank fusion
 generation.py          Streaming, prompts, citations and CV evidence validation
 cv.py                  PDF text extraction and local OCR
 ollama.py              Local model checks and safe backend errors
 config.py              Paths and environment settings
 cli.py                 Ingest/build/search/ask commands
 evaluation.py          Retrieval and generation experiments
data/                   Fixed JobTech snapshot
evaluation/             Draft questions, annotation guide and measured results
docs/                   Report draft, preliminary results and original project description
tests/                  Pipeline, interface, OCR and local-model tests
```

Local runtime directories `.venv/`, `.cache/` and `storage/` are generated and excluded from submission. Keep them locally to avoid reinstalling dependencies, downloading models and rebuilding the active index.

## Data and retrieval

To refresh the dataset (this changes the experimental snapshot):

```bash
python -m jobmatch.cli ingest --scope data-science --limit 3000
python -m jobmatch.cli build
python -m jobmatch.cli search "machine learning Python" --location Stockholm
python -m jobmatch.cli ask "Which roles mention SQL?"
```

Ingestion uses JobTech `/v2/snapshot` and a Swedish/English keyword filter. It can include adjacent roles or miss synonyms. Cleaning removes empty/duplicate ads. Retrieval uses `intfloat/multilingual-e5-small` with `passage:`/`query:` prefixes, 320-token chunks with 60 overlap, persistent Chroma, and BM25L. Hybrid ranking combines both lists with reciprocal-rank fusion and returns at most two chunks per ad. Long queries/CVs use averaged segment embeddings. Similarity thresholds are experimental, not probabilities.

Generation uses retrieved evidence and numbered citations. Empty retrieval refuses without calling a model. CV comparison validates literal source/CV quotes before displaying related wording; it does not certify proficiency, completed education or eligibility. Translations, completeness and claim support still require checking. Ads may expire or change after the snapshot.

## Evaluation and tests

```bash
python -m pytest -q
python -m jobmatch.evaluation run --output evaluation/results/baseline
# Optional: invokes the selected model for RAG/no-RAG comparisons
python -m jobmatch.evaluation run --generation --limit 3 --output evaluation/results/openai
python -m jobmatch.evaluation run --generation --backend ollama --model qwen2.5:7b --output evaluation/results/qwen
```

The 30 questions in `evaluation/questions.json` are a **draft**, not a reviewed benchmark. Review expected facts, CV gold labels and paraphrases using `evaluation/ANNOTATION.md`; set `reviewed: true` only after checking sources. Exact-title questions make retrieval artificially easy.

Recorded preliminary retrieval results for 200/320/400-token chunks are preserved in `evaluation/results/`. See [preliminary results](docs/preliminary_results.md). Alternate indexes can be rebuilt:

```bash
python -m jobmatch.cli build --chunk-size 200 --overlap 40 --index storage/chunks200
python -m jobmatch.evaluation run --index storage/chunks200 --output evaluation/results/chunks200
```

Generation CSVs record answers, sources, token usage and latency. Manually score supported/unsupported claims and correct refusals; numeric citation validity alone is not faithfulness. Supply `--input-rate`/`--output-rate` in USD per million tokens to estimate API cost. Use synthetic CVs in shared results.

## Submission

`submission.zip` contains source, pinned requirements, tests, documentation, the data snapshot and recorded retrieval results. It excludes `.env`, local models, environments, indexes, caches and editor settings. The report in `docs/report.md` is still a draft: review labels, finish generation scoring and replace its placeholders before submitting the final report. Declare AI assistance and actual individual contributions truthfully.

## References

- [JobTech JobStream](https://jobstream.api.jobtechdev.se/swagger.json) — source ads, CC0.
- [Multilingual E5](https://huggingface.co/intfloat/multilingual-e5-small) — model and retrieval prefixes.
- [OpenAI Responses](https://developers.openai.com/api/docs/guides/streaming-responses) — generation API.
- [Chroma](https://docs.trychroma.com/) — vector storage.
