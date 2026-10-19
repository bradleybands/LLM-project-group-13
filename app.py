"""Run with: streamlit run app.py"""
import hashlib
import os
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from jobmatch.config import INDEX_PATH, OPENAI_MODEL
from jobmatch.cv import CVExtractionError, extract_cv_pdf
from jobmatch.data import is_expired
from jobmatch.generation import GenerationStats, stream_answer, is_cv_match_question
from jobmatch.ollama import LocalModelError, installed_models
from jobmatch.retrieval import Retriever

st.set_page_config(page_title="JobMatch · Your career workspace", page_icon="🔎", layout="wide")
st.markdown("""<style>
.stApp, input, textarea, button, h1, h2, h3, p {
 font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',Arial,sans-serif!important;}
.block-container {max-width:1040px;padding-top:2.2rem;padding-bottom:3rem;}
h1,h2,h3 {letter-spacing:-.045em;font-weight:600!important;color:#1d1d1f;}
[data-testid="stSidebar"] {background:#f5f5f7;border-right:1px solid #e5e5ea;}
[data-testid="stChatMessage"] {background:transparent;border:0;padding:1.2rem 0;}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {line-height:1.7;}
[data-testid="stChatInput"] {border:1px solid #e5e5ea;border-radius:22px;background:#fff;
 box-shadow:0 8px 32px #00000008;}
[data-testid="stBottom"] {background:#ffffffed;backdrop-filter:blur(20px);}
.welcome {text-align:center;padding:8vh 0 2rem;}
.welcome .eyebrow {color:#0071e3;font-size:13px;font-weight:600;letter-spacing:.08em;
 text-transform:uppercase;margin-bottom:18px;}
.welcome h1 {font-size:clamp(36px,4vw,56px);line-height:1.08;margin:0 0 22px;}
.welcome p {color:#86868b;font-size:19px;max-width:540px;margin:auto;line-height:1.55;}
.brand {font-size:22px;font-weight:600;letter-spacing:-.8px;margin:0 0 22px;color:#1d1d1f;}
button {border-radius:14px!important;transition:background .15s ease,box-shadow .15s ease;}
[data-testid="stBaseButton-secondary"] {background:#fff;border-color:#e5e5ea;}
[data-testid="stBaseButton-secondary"]:hover {border-color:#c7c7cc;box-shadow:0 3px 12px #00000006;}
[data-testid="stExpander"] {border-color:#e5e5ea;border-radius:14px;}
@media(max-width:640px) {.welcome {padding-top:3vh}.block-container{padding-top:1rem}}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_retriever(path: str, revision: str):
    return Retriever(Path(path))


def safe_url(url: str) -> bool:
    return urlparse(url).scheme in {"http", "https"}


def show_sources(chunks: list[dict], cited: list[int] | None = None):
    if not chunks:
        return
    with st.expander(f"Sources · {len(chunks)} excerpts", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            marker = " · cited" if cited and i in cited else ""
            st.markdown(f"**[{i}] {meta['headline']}{marker}**")
            st.caption(f"{meta['employer']} · {meta['location']} · {meta['role']}")
            st.caption(f"Published: {meta['published'] or 'not specified'} · Deadline: {meta['deadline'] or 'not specified'}")
            st.text(chunk["text"])
            if safe_url(meta["url"]):
                st.link_button("Open original ad", meta["url"])
            if is_expired(meta):
                st.caption("The listed deadline has passed.")
            st.divider()


def clear_chat():
    st.session_state["messages"] = []


def reset_consent():
    st.session_state["cv_consent"] = False
    clear_chat()


def reset_uploaded_cv():
    st.session_state["cv"] = ""
    st.session_state["cv_signature"] = None
    st.session_state["cv_extraction"] = None
    st.session_state["cv_error"] = None
    reset_consent()


def clear_everything():
    reset_uploaded_cv()
    st.session_state["upload_revision"] = st.session_state.get("upload_revision", 0) + 1


for key, value in {"messages": [], "cv": "", "cv_consent": False, "upload_revision": 0,
                   "cv_signature": None, "cv_extraction": None, "cv_error": None}.items():
    if key not in st.session_state:
        st.session_state[key] = value

manifest_path = INDEX_PATH / "manifest.json"
if not manifest_path.exists():
    st.title("JobMatch")
    st.info("Build your searchable dataset first. No OpenAI key is needed for these steps.")
    st.code(".venv/bin/python -m jobmatch.cli ingest\n.venv/bin/python -m jobmatch.cli build", language="bash")
    st.stop()
try:
    retriever = get_retriever(str(INDEX_PATH), manifest_path.read_text())
except Exception:
    st.error("The local index could not be loaded. Rebuild it with python -m jobmatch.cli build.")
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand">JobMatch</div>', unsafe_allow_html=True)
    st.button("＋ New chat", on_click=clear_chat, width="stretch")
    view = st.radio("Workspace", ["Chat", "Job library"], label_visibility="collapsed")
    st.divider()
    with st.expander("Model settings", expanded=False):
        backend_label = st.selectbox("Model provider", ["OpenAI", "Ollama (local)"],
                                     key="provider", on_change=reset_consent)
        backend = "openai" if backend_label == "OpenAI" else "ollama"
        model = st.text_input("Model", key=f"model_{backend}",
                             value=OPENAI_MODEL if backend == "openai" else os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))
        api_key = ""
        if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
            api_key = st.text_input("OpenAI API key", type="password", help="Session only. You can also set OPENAI_API_KEY in .env.")
        if backend == "ollama":
            try:
                available_models = installed_models()
                if available_models:
                    st.caption("Installed locally: " + ", ".join(available_models))
                else:
                    st.warning("Ollama is running, but no models are downloaded. Run ollama pull qwen2.5:7b.")
            except LocalModelError as exc:
                st.warning(str(exc))
    locations = sorted({a["location"] for a in retriever.ads})
    roles = sorted({a["role"] for a in retriever.ads})
    with st.expander("Search filters", expanded=False):
        location = st.selectbox("Location", ["All locations"] + locations)
        role = st.selectbox("Occupation", ["All occupations"] + roles)
        mode = st.selectbox("Retrieval", ["hybrid", "semantic", "keyword"])
        k = st.slider("Source excerpts per answer", 2, 12, 6)
        similarity = st.slider("Minimum semantic similarity", 0.60, 0.95, 0.72, 0.01,
                               help="Experimental threshold, not a probability. Hybrid can still return keyword matches.")
    with st.expander("Your CV · PDF or text", expanded=True):
        uploaded = st.file_uploader("Upload CV (PDF)", type=["pdf"],
                                    key=f"cv_pdf_{st.session_state['upload_revision']}",
                                    on_change=reset_uploaded_cv,
                                    help="Up to 10 MB / 12 pages. Text extraction and OCR happen locally.")
        force_ocr = st.checkbox("Force OCR on all pages", help="Use if a PDF's existing text is missing or garbled.")
        if uploaded is not None:
            data = uploaded.getvalue()
            signature = (hashlib.sha256(data).hexdigest(), force_ocr)
            if signature != st.session_state["cv_signature"]:
                reset_consent()
                st.session_state["cv"] = ""
                st.session_state["cv_extraction"] = None
                st.session_state["cv_error"] = None
                try:
                    with st.spinner("Reading your PDF…"):
                        extraction = extract_cv_pdf(data, force_ocr=force_ocr)
                    st.session_state["cv"] = extraction.text
                    st.session_state["cv_extraction"] = extraction
                except CVExtractionError as exc:
                    st.session_state["cv_error"] = str(exc)
                except Exception:
                    st.session_state["cv_error"] = "The PDF could not be read. Try a fresh PDF or paste the text."
                st.session_state["cv_signature"] = signature
            if st.session_state["cv_error"]:
                st.error(st.session_state["cv_error"])
            extraction = st.session_state["cv_extraction"]
            if extraction:
                st.success(f"Read {extraction.page_count} page(s) · OCR on {len(extraction.ocr_pages)}")
                for warning in extraction.warnings:
                    st.warning(warning)
        st.text_area("CV text", key="cv", height=210, max_chars=16000, on_change=reset_consent,
                     placeholder="Upload a PDF or paste your experience and skills…",
                     help="Review extraction/OCR errors and remove personal contact details before sharing.")
        consent_label = "Send my reviewed CV text to OpenAI" if backend == "openai" else "Use my reviewed CV with local Ollama"
        st.checkbox(consent_label, key="cv_consent")
        st.caption("The PDF stays local. Only the reviewed text is sent to your selected model with consent. CVs are not saved to disk or indexed.")
        cv_clicked = st.button("Find matching roles", type="primary", width="stretch",
                               disabled=not st.session_state["cv"].strip() or not st.session_state["cv_consent"])
    with st.expander("Workspace details"):
        st.caption(f"{len(retriever.ads)} ads · {len(locations)} locations\n\nSnapshot: {retriever.manifest['built_at'][:10]}")
        st.caption("Original ads are Swedish or English; answers are English. Check original listings before applying.")
        st.button("Clear conversation and CV", on_click=clear_everything, width="stretch")

filters = {"location": None if location == "All locations" else location,
           "role": None if role == "All occupations" else role}
filtered_ads = retriever.filter_ads(filters)
st.caption("JOBMATCH  /  YOUR CAREER WORKSPACE")

if view == "Job library" and not cv_clicked:
    st.title("Your job library")
    st.caption(f"{len(filtered_ads)} ads matching your filters. Counts describe this snapshot, not the entire job market.")
    search_text = st.text_input("Find a title, company or skill")
    visible = [a for a in filtered_ads if search_text.casefold() in (a["headline"] + " " + a["employer"] + " " + a["text"]).casefold()]
    st.dataframe([{key: a[key] for key in ("headline", "employer", "location", "role", "deadline", "url")} for a in visible],
                 width="stretch", hide_index=True, column_config={"url": st.column_config.LinkColumn("Original ad")})
    if visible:
        selected = st.selectbox("Read an original ad", range(len(visible)),
                                format_func=lambda i: f"{visible[i]['headline']} — {visible[i]['employer']}")
        st.text(visible[selected]["text"])
    st.stop()

quick_question = None
if not st.session_state["messages"]:
    st.markdown('''<div class="welcome"><div class="eyebrow">Your next opportunity</div>
    <h1>A clearer path.<br>A better next step.</h1>
    <p>Explore Swedish data and tech jobs, understand the requirements,
    and find where your experience fits.</p></div>''', unsafe_allow_html=True)
    suggestions = st.columns(3)
    if suggestions[0].button("Explore ML roles", width="stretch"):
        quick_question = "Which machine learning roles mention Python, and what requirements do they state?"
    if suggestions[1].button("Check language requirements", width="stretch"):
        quick_question = "What language requirements are explicitly stated in these data science job ads?"
    if suggestions[2].button("Compare my CV", width="stretch"):
        if st.session_state["cv"].strip() and st.session_state["cv_consent"]:
            cv_clicked = True
        else:
            st.info("Upload or paste your CV in the sidebar, review the text, and enable consent to compare it.")
    st.caption("Answers grounded in job ads, with sources you can inspect.")
else:
    st.markdown("### Your conversation")
if st.session_state["cv"].strip():
    state = "ready to use" if st.session_state["cv_consent"] else "waiting for your review and consent"
    st.caption(f"CV attached · {state}")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"], avatar=":material/work_outline:" if message["role"] == "assistant" else ":material/person_outline:"):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources", []), message.get("stats", {}).get("citation_ids"))

question = st.chat_input("Ask about jobs, skills, or your next move…", max_chars=6000) or quick_question
if cv_clicked:
    question = "Which retrieved roles fit my CV? Compare the stated requirements against my CV evidence, including language requirements."
if question:
    cv = st.session_state["cv"] if st.session_state["cv_consent"] else ""
    if st.session_state["cv"].strip() and not st.session_state["cv_consent"] and is_cv_match_question(question):
        st.warning(f'Your CV is attached but not enabled for this provider. Review the text and tick "{consent_label}" in the sidebar, then ask again.')
    elif backend == "openai" and not (api_key or os.getenv("OPENAI_API_KEY")):
        st.warning("Enter your OpenAI API key in Model settings or add it to .env, then restart the app.")
    elif not model.strip():
        st.warning("Enter a model name.")
    else:
        history = list(st.session_state["messages"])
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user", avatar=":material/person_outline:"):
            st.markdown(question)
        with st.chat_message("assistant", avatar=":material/work_outline:"):
            try:
                with st.spinner("Finding relevant job ads…"):
                    prior_queries = [m["content"] for m in history if m["role"] == "user"]
                    retrieval_query = question
                    if prior_queries and not cv_clicked:
                        retrieval_query += "\nPrevious question: " + prior_queries[-1][:1000]
                    if cv:
                        retrieval_query += "\nCV skills and experience: " + cv
                    chunks = retriever.retrieve(retrieval_query, k, filters, mode, similarity)
                stats = GenerationStats()
                answer = st.write_stream(stream_answer(question, chunks, model=model.strip(), backend=backend,
                                                       cv=cv, history=history, api_key=api_key or None, stats=stats))
                if stats.invalid_citation_ids:
                    st.warning("The answer contains an invalid reference. Verify its claims against the excerpts.")
                elif chunks and not stats.citation_ids:
                    st.warning("This answer has no source citations. Treat job claims as unverified.")
                show_sources(chunks, stats.citation_ids)
                st.session_state["messages"].append({"role": "assistant", "content": answer,
                                                    "sources": chunks, "stats": asdict(stats)})
            except LocalModelError as exc:
                st.error(str(exc))
                st.session_state["messages"].pop()
            except Exception as exc:
                st.error(f"Request failed ({type(exc).__name__}). Check the model settings and connection, then try again.")
                st.session_state["messages"].pop()
        if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "assistant":
            st.rerun()
