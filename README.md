# employmee

*ending unemployment*

A chat assistant for Swedish tech jobs and master's thesis opportunities: AI/ML, data, software and computer engineering, systems architecture, cloud, networks and security. It searches real JobTech ads and answers in English with source citations. You can also upload a PDF CV to compare your experience with job requirements.

## Setup

Use Python 3.11 or newer:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env
```

For OpenAI, add your API key to `OPENAI_API_KEY` in `.env`. Keep this file private.
The default model is [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), using low reasoning effort. You can change the model in the sidebar or with `OPENAI_MODEL` in `.env`.

## Run the app

Start the app:

```bash
./run.sh
```

Open **http://127.0.0.1:8501**. On page load, the app checks the time of the last successful JobTech update. If it was less than **6 hours ago**, it reuses the saved database without downloading or rebuilding. Otherwise, it downloads current tech ads and updates new, changed and removed listings. Chat messages and sidebar changes do not trigger another check.

The last successful update time is saved on disk, so the six-hour window survives page reloads and app restarts. A successful check resets this time even if the ads are unchanged; a failed update does not. **Workspace details** shows the time in Europe/Stockholm.

The app builds the index automatically and skips rebuilding when the job data is unchanged. The first load also downloads the embedding model and can take longer. No OpenAI key is needed for updating jobs. If an update fails, the app warns you and uses the last working database.

The tech filter uses JobTech's Data/IT category and technical role titles. For thesis ads with generic titles, it also checks the project description for technical topics. It recognises English thesis titles and Swedish titles such as **Examensarbete** and **Exjobb**. Changes to this filter trigger one fresh download even inside the six-hour window. The downloaded ads are saved locally in `data/tech_jobs.jsonl` and excluded from Git. Restart the app after changing code or `.env` settings.

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

## Chatting

Ask about jobs, skills, technical learning, companies, working life, applications or CV edits. Unrelated topics receive a brief redirect. Career advice does not trigger a job search.
Ask “Find master's thesis opportunities in machine learning” or select **Thesis opportunities** under **Opportunity type** to browse them. Only openings published on JobTech are available; check each original ad for its degree level and eligibility.
Current date and time come from the app's system clock in Europe/Stockholm. Follow-up questions refer to the last answer and its sources.
Once you enable CV consent, the reviewed CV stays available throughout the current chat, including indirect skills questions and follow-ups. Chat history and your CV are held in the current app session; they are not saved to disk.

For a specific role, choose it under **Focus on a specific job**, name the company/job, or refer to an earlier result such as “the second job” or “[2]”. Then ask “How can I tailor my CV for this role?” or “Rewrite my project bullet.” Follow-ups such as “Make it shorter” keep the conversation context. Tailoring uses that role's requirements and checks proposed CV wording against your original text.

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

