"""Run with: streamlit run app.py"""
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import streamlit as st

from jobmatch.config import INDEX_PATH, OPENAI_MODEL
from jobmatch.cv import CVExtractionError, extract_cv_pdf
from jobmatch.data import is_expired
from jobmatch.ad_display import original_ad_html
from jobmatch.generation import GenerationStats, stream_answer
from jobmatch.ollama import LocalModelError, installed_models
from jobmatch.conversation import intent_for, resolve_job, focused_sources, latest_sources, previous_answer, active_request, uses_cv, thesis_opportunity_query
from jobmatch.runtime import APP_TIMEZONE, is_clock_followup
from jobmatch.retrieval import Retriever
from jobmatch.ingest import refresh_jobs, last_job_update

st.set_page_config(page_title="EmployMee · ending unemployment", page_icon="✦", layout="wide")
style = (Path(__file__).parent / ".streamlit" / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


@st.cache_resource(max_entries=2)
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


def new_chat():
    clear_chat()
    st.session_state["target_ad_id"] = None


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
    st.session_state["target_ad_id"] = None
    st.session_state["upload_revision"] = st.session_state.get("upload_revision", 0) + 1


for key, value in {"messages": [], "cv": "", "cv_consent": False, "upload_revision": 0,
                   "cv_signature": None, "cv_extraction": None, "cv_error": None}.items():
    if key not in st.session_state:
        st.session_state[key] = value
if "pending_target_id" in st.session_state:
    st.session_state["target_ad_id"] = st.session_state.pop("pending_target_id")

manifest_path = INDEX_PATH / "manifest.json"
if "jobs_refresh_attempted" not in st.session_state:
    # A new page/browser session checks the persistent six-hour cache once.
    # Streamlit also reruns this file
    # for messages and widget edits, which must not trigger another download.
    st.session_state["jobs_refresh_attempted"] = True
    with st.spinner("Checking tech jobs… Older databases are updated from JobTech."):
        try:
            refresh_jobs(INDEX_PATH)
        except Exception:
            logging.getLogger(__name__).exception("Automatic JobTech update failed")
            st.session_state["jobs_refresh_failed"] = True
        else:
            st.session_state["jobs_refresh_failed"] = False
if st.session_state.get("jobs_refresh_failed"):
    if manifest_path.exists():
        st.warning("Jobs could not be updated. Using the last saved database; some listings may have changed.")
    else:
        st.error("Jobs could not be downloaded. Check your connection and try again.")
    if st.button("Retry jobs update"):
        st.session_state.pop("jobs_refresh_attempted", None)
        st.rerun()
if not manifest_path.exists():
    st.title("employmee")
    st.stop()
try:
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    # Checking an unchanged feed should not create another cached model/client.
    retriever = get_retriever(str(INDEX_PATH), manifest.get("generation", manifest_text))
except Exception:
    st.error("The local index could not be loaded. Rebuild it with python -m jobmatch.cli build.")
    st.stop()

with st.sidebar:
    st.markdown('''<div class="brand"><div class="brand-mark" aria-hidden="true">em.</div>
    <div><div class="brand-name">employmee</div><div class="brand-tagline">ending unemployment</div></div></div>''', unsafe_allow_html=True)
    st.button("＋ New chat", on_click=new_chat, width="stretch")
    view = st.radio("Workspace", ["Chat", "Job library"], label_visibility="collapsed", key="workspace_view", horizontal=True)
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
        opportunity = st.selectbox("Opportunity type", ["All opportunities", "Jobs", "Thesis opportunities"])
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
        st.text_area("CV text", key="cv", height=140, max_chars=16000, on_change=reset_consent,
                     placeholder="Upload a PDF or paste your experience and skills…",
                     help="Review extraction/OCR errors and remove personal contact details before sharing.")
        consent_label = "Send my reviewed CV text to OpenAI" if backend == "openai" else "Use my reviewed CV with local Ollama"
        st.checkbox(consent_label, key="cv_consent")
        st.caption("The PDF stays local. Only the reviewed text is sent to your selected model with consent. CVs are not saved to disk or indexed.")
        cv_clicked = st.button("Find matching roles", type="primary", width="stretch",
                               disabled=not st.session_state["cv"].strip() or not st.session_state["cv_consent"])
    with st.expander("Workspace details"):
        checked_at = last_job_update(manifest)
        update_label = (f"Last successful update: {checked_at.astimezone(ZoneInfo(APP_TIMEZONE)):%Y-%m-%d %H:%M} ({APP_TIMEZONE})"
                        if checked_at else f"Saved index built: {retriever.manifest['built_at'][:10]} · refresh unavailable")
        st.caption(f"{len(retriever.ads)} tech ads · {len(locations)} locations\n\n{update_label}")
        st.caption("On page load, the saved database is reused if its last successful update was less than 6 hours ago.")
        st.caption("Original ads are Swedish or English; answers are English. Check original listings before applying.")
        st.button("Clear conversation and CV", on_click=clear_everything, width="stretch")

filters = {"location": None if location == "All locations" else location,
           "role": None if role == "All occupations" else role,
           "opportunity_type": {"Jobs": "job", "Thesis opportunities": "thesis"}.get(opportunity)}
filtered_ads = retriever.filter_ads(filters)

if view == "Job library" and not cv_clicked:
    st.markdown('<div class="library-kicker">GOOD THINGS ARE OUT THERE</div>', unsafe_allow_html=True)
    st.title("The opportunity shelf.")
    st.caption(f"{len(filtered_ads):,} jobs and thesis opportunities matching your filters. Check original listings before applying.")
    search_text = st.text_input("Find a title, company or skill")
    visible = [a for a in filtered_ads if search_text.casefold() in (a["headline"] + " " + a["employer"] + " " + a["text"]).casefold()]
    st.dataframe([{key: a[key] for key in ("headline", "employer", "location", "role", "deadline", "url")} for a in visible],
                 width="stretch", hide_index=True, column_config={"url": st.column_config.LinkColumn("Original ad")})
    if visible:
        selected = st.selectbox("Read an original ad", range(len(visible)),
                                format_func=lambda i: f"{visible[i]['headline']} — {visible[i]['employer']}")
        ad = visible[selected]
        st.subheader(ad["headline"])
        st.caption(f"{ad['employer']} · {ad['location']} · {ad['role']}")
        st.caption(f"Published: {ad['published'] or 'not specified'} · Deadline: {ad['deadline'] or 'not specified'}")
        if safe_url(ad["url"]):
            st.link_button("Open original listing", ad["url"])
        if is_expired(ad):
            st.caption("The listed deadline has passed.")
        st.html('<article class="original-ad">' + original_ad_html(ad, retriever.manifest.get("source")) + '</article>')
    st.stop()

ads_by_id = {ad["ad_id"]: ad for ad in retriever.ads}
if st.session_state.get("target_ad_id") not in ads_by_id:
    st.session_state["target_ad_id"] = None
quick_question = None
if not st.session_state["messages"]:
    st.markdown('''<div class="welcome"><div class="eyebrow">Big career energy</div>
    <div class="hero-sticker">one application<br><strong>at a time ↗</strong></div>
    <h1><span class="hero-first-line">ending
    <svg class="hero-spark" viewBox="0 0 80 80" aria-hidden="true"><path d="M40 3 48 25 70 13 58 35 79 43 56 50 66 73 44 60 35 80 28 55 5 66 18 44 0 35 24 28 15 7 35 21Z" fill="#dff3ae" stroke="#7136df" stroke-width="2.5" stroke-linejoin="round"/></svg>
    </span><br><span class="hero-word">unemployment</span></h1>
    <p>Less doomscrolling. More doors opening.<br>
    Find tech jobs, thesis projects and a next step that feels like you.</p></div>''', unsafe_allow_html=True)
    cv_help_needed = False
    with st.container(key="launch_cards"):
        suggestions = st.columns(3)
        with suggestions[0]:
            if st.button("Explore ML roles", icon=":material/explore:", width="stretch"):
                quick_question = "Which machine learning roles mention Python, and what requirements do they state?"
            st.caption("Find your kind of tech.")
        with suggestions[1]:
            if st.button("Find a thesis", icon=":material/school:", width="stretch"):
                quick_question = "Find master's thesis opportunities in machine learning and data science."
            st.caption("Big ideas. A place to start.")
        with suggestions[2]:
            if st.button("Give my CV a glow-up", icon=":material/auto_awesome:", width="stretch"):
                if st.session_state["cv"].strip() and st.session_state["cv_consent"]:
                    quick_question = "Review my CV and suggest three concrete improvements based on my actual experience."
                else:
                    cv_help_needed = True
            st.caption("Your experience, better told.")
    if cv_help_needed:
        st.info("Upload or paste your CV in the sidebar, review the text, and enable consent to get started.")
    st.markdown('<div class="trust-note"><span aria-hidden="true">✦</span> Real JobTech ads. Sources you can check. Advice you can act on.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="conversation-title">Let’s make moves.</div>', unsafe_allow_html=True)

with st.expander("Focus on a specific job", expanded=False):
    target_id = st.selectbox(
        "Target job (optional)", [None] + list(ads_by_id), key="target_ad_id",
        format_func=lambda value: "No target selected" if value is None else
        f"{ads_by_id[value]['headline']} — {ads_by_id[value]['employer']} · {ads_by_id[value]['location']}",
        help="Choose the role for CV tailoring, rewriting, or questions about one job. This target takes precedence over broad search filters.",
    )
if target_id:
    target_ad = ads_by_id[target_id]
    st.caption(f"Target: {target_ad['headline']} · {target_ad['employer']}")
if st.session_state["cv"].strip():
    state = "ready to use" if st.session_state["cv_consent"] else "waiting for your review and consent"
    st.caption(f"CV attached · {state}")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"], avatar=":material/work_outline:" if message["role"] == "assistant" else ":material/person_outline:"):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources", []), message.get("stats", {}).get("citation_ids"))

question = st.chat_input("Dream job? Thesis idea? CV wobble? Let’s talk…", max_chars=6000) or quick_question
if cv_clicked:
    question = "Which retrieved roles fit my CV? Compare the stated requirements against my CV evidence, including language requirements."
if question:
    history = list(st.session_state["messages"])
    task = intent_for(question, history)
    local_task = task == "clock" or (task == "follow_up" and is_clock_followup(history))
    personal_task = uses_cv(question, history)
    request = active_request(question, history, task)
    # Consent applies to this provider's conversation, not a keyword classifier.
    # Always supply the reviewed CV so indirect questions cannot lose it.
    cv = st.session_state["cv"] if st.session_state["cv_consent"] else ""
    if st.session_state["cv"].strip() and not st.session_state["cv_consent"] and personal_task:
        st.warning(f'Your CV is attached but not enabled for this provider. Review the text and tick "{consent_label}" in the sidebar, then ask again.')
    elif not local_task and backend == "openai" and not (api_key or os.getenv("OPENAI_API_KEY")):
        st.warning("Enter your OpenAI API key in Model settings or add it to .env, then restart the app.")
    elif not local_task and not model.strip():
        st.warning("Enter a model name.")
    else:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user", avatar=":material/person_outline:"):
            st.markdown(question)
        with st.chat_message("assistant", avatar=":material/work_outline:"):
            try:
                with st.spinner("Preparing your answer…"):
                    resolved_id = resolve_job(question, retriever.ads, history, target_id)
                    chunks = []
                    if task == "follow_up" and not local_task:
                        chunks = previous_answer(history).get("sources", [])
                    elif resolved_id and task in {"tailor", "rewrite", "job_question", "cv_match"}:
                        chunks = focused_sources(retriever.ads, resolved_id)
                        st.session_state["pending_target_id"] = resolved_id
                    elif task == "job_question" and latest_sources(history) and re.search(r"\b(?:these|those|them|they|their)\b", question, re.IGNORECASE):
                        chunks = latest_sources(history)
                    elif task in {"job_search", "job_question", "cv_match"}:
                        retrieval_query = question
                        if task == "cv_match" and cv:
                            retrieval_query += "\nCV skills and experience: " + cv
                        search_filters = dict(filters)
                        if task in {"job_search", "cv_match"} and thesis_opportunity_query(question):
                            search_filters["opportunity_type"] = "thesis"
                        chunks = retriever.retrieve(retrieval_query, k, search_filters, mode, similarity)
                stats = GenerationStats()
                answer = st.write_stream(stream_answer(question, chunks, model=model.strip(), backend=backend,
                                                       cv=cv, history=history, api_key=api_key or None, stats=stats,
                                                       intent=task))
                if stats.answer_origin == "scope_guard":
                    # A rejected topic must not show or carry forward unrelated
                    # ads that a broad task router may have retrieved for it.
                    chunks = []
                    task = "out_of_scope"
                    request = question
                if stats.invalid_citation_ids:
                    st.warning("The answer contains an invalid reference. Verify its claims against the excerpts.")
                elif chunks and not stats.citation_ids and task in {"job_search", "job_question", "cv_match"}:
                    st.warning("This answer has no source citations. Treat job claims as unverified.")
                show_sources(chunks, stats.citation_ids)
                st.session_state["messages"].append({"role": "assistant", "content": answer,
                                                    "sources": chunks, "stats": asdict(stats), "intent": task,
                                                    "active_request": request})
            except LocalModelError as exc:
                st.error(str(exc))
                st.session_state["messages"].pop()
            except Exception as exc:
                st.error(f"Request failed ({type(exc).__name__}). Check the model settings and connection, then try again.")
                st.session_state["messages"].pop()
        if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "assistant":
            st.rerun()
