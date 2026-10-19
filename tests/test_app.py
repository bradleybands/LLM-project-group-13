"""UI regression checks use local fake retrieval and no external API calls."""
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from jobmatch.ingest import refresh_jobs as refresh_saved_jobs, TECH_FILTER_VERSION


@pytest.fixture(autouse=True)
def jobs_update():
    # UI tests must never download public data or run an embedding model.
    with patch("jobmatch.ingest.refresh_jobs") as update:
        yield update


def test_ui_cv_consent_sources_and_provider_change(monkeypatch, tmp_path):
    import jobmatch.config
    # Load .env first so this missing-key scenario also works in isolation.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(jobmatch.config, "INDEX_PATH", tmp_path)
    (tmp_path / "manifest.json").write_text('{}')
    ad = {"ad_id": "1", "headline": "Data Scientist", "text": "SQL required", "location": "Uppsala",
          "role": "Analyst", "employer": "Example", "deadline": "", "published": "",
          "url": "https://example.com/ad"}
    chunk = {"chunk_id": "1:0", "text": "SQL required", "metadata": {k: v for k, v in ad.items() if k != "text"}}
    fake = SimpleNamespace(ads=[ad], manifest={"built_at": "2026-10-06"},
                           filter_ads=lambda filters: [ad], retrieve=lambda *a, **kw: [chunk])
    def answer(*args, **kwargs):
        kwargs["stats"].citation_ids = [1]
        yield "The role requires SQL [1]."
    with patch("jobmatch.retrieval.Retriever", return_value=fake), patch("jobmatch.generation.stream_answer", side_effect=answer):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
        assert not app.exception

        # A missing key must stop before a model request.
        app.chat_input[0].set_value("What skills are required?").run()
        assert app.warning and not app.session_state["messages"]
        key_field = next(t for t in app.text_input if t.label == "OpenAI API key")
        key_field.set_value("test-key-not-real").run()
        app.chat_input[0].set_value("What skills are required?").run()
        assert not app.exception
        assert len(app.session_state["messages"]) == 2
        assert app.session_state["messages"][-1]["sources"][0]["metadata"]["ad_id"] == "1"
        # A tailoring turn must focus the selected job and pass the task through,
        # rather than performing a fresh broad search or producing a JSON table.
        app.text_area(key="cv").set_value("SQL projects").run()
        app.checkbox(key="cv_consent").check().run()
        app.selectbox(key="target_ad_id").set_value("1").run()
        with patch.object(fake, "retrieve", side_effect=AssertionError("Unexpected broad search")):
            app.chat_input[0].set_value("How can I tailor my CV for this role?").run()
        assert not app.exception
        assert app.session_state["messages"][-1]["sources"][0]["chunk_id"] == "1:focused"
        with patch.object(fake, "retrieve", side_effect=AssertionError("Follow-up must reuse its answer's sources")):
            app.chat_input[0].set_value("How do you know this?").run()
        assert not app.exception
        assert app.session_state["messages"][-1]["sources"][0]["chunk_id"] == "1:focused"
        # General advice should not retrieve ads, even while a target is selected.
        with patch.object(fake, "retrieve", side_effect=AssertionError("Advice must not search")):
            app.chat_input[0].set_value("How do I structure a CV?").run()
        assert not app.exception
        assert app.session_state["messages"][-1]["sources"] == []
        with patch.object(fake, "retrieve", side_effect=AssertionError("Follow-up must not search")):
            app.chat_input[0].set_value("How do you know this?").run()
        assert not app.exception
        assert app.session_state["messages"][-1]["sources"] == []
        # Editing the CV resets consent before another personalized question.
        app.text_area(key="cv").set_value("SQL and Python projects").run()
        app.chat_input[0].set_value("what jobs can i get with my experience").run()
        assert not app.session_state["messages"]
        assert any("CV is attached but not enabled" in warning.value for warning in app.warning)
        app.checkbox(key="cv_consent").check().run()
        app.selectbox(key="provider").select("Ollama (local)").run()
        assert not app.checkbox(key="cv_consent").value
        next(b for b in app.button if b.label == "Clear conversation and CV").click().run()
        assert not app.session_state["messages"] and app.session_state["cv"] == ""
        assert not app.exception


def test_clock_and_short_followups_ignore_selected_jobs_and_need_no_api_key(monkeypatch, tmp_path):
    from datetime import datetime
    from jobmatch.runtime import runtime_context
    clock = runtime_context(datetime.fromisoformat("2026-10-19T14:30:00+00:00"))
    import jobmatch.config
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(jobmatch.config, "INDEX_PATH", tmp_path)
    (tmp_path / "manifest.json").write_text('{}')
    ad = {"ad_id": "nordea", "headline": "AI Scientist", "text": "SQL required", "location": "Stockholm",
          "role": "Scientist", "employer": "Nordea", "deadline": "", "published": "2026-10-05",
          "url": "https://example.com/ad"}
    fake = SimpleNamespace(ads=[ad], manifest={"built_at": "2026-10-06"}, filter_ads=lambda filters: [ad])
    with patch("jobmatch.retrieval.Retriever", return_value=fake), \
         patch("jobmatch.generation.runtime_context", return_value=clock), \
         patch("openai.OpenAI", side_effect=AssertionError("Unexpected model call")):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
        app.text_area(key="cv").set_value("SQL projects").run()
        app.selectbox(key="target_ad_id").set_value("nordea").run()
        # Simulate an already open conversation with unrelated job citations.
        app.session_state["messages"] = [
            {"role": "user", "content": "When was the Nordea ad listed?"},
            {"role": "assistant", "content": "5 October 2026 [1].", "sources": []},
            {"role": "user", "content": "what is today's date"},
            {"role": "assistant", "content": "June 25, 2024.", "sources": []}]
        for question in ["how do you know this", "so what is today's date", "how"]:
            app.chat_input[0].set_value(question).run()
            assert not app.exception and not app.error
            answer = app.session_state["messages"][-1]
            assert "19 October 2026" in answer["content"]
            assert answer["sources"] == []
            assert answer["stats"]["answer_origin"] == "runtime_clock"


def test_reviewed_cv_is_retained_for_indirect_questions_and_corrections(monkeypatch, tmp_path):
    import jobmatch.config
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(jobmatch.config, "INDEX_PATH", tmp_path)
    (tmp_path / "manifest.json").write_text('{}')
    ad = {"ad_id": "1", "headline": "Data Scientist", "text": "SQL required", "location": "Uppsala",
          "role": "Analyst", "employer": "Example", "deadline": "", "published": "",
          "url": "https://example.com/ad"}
    # No retrieve method: these requests must never start an ad search.
    fake = SimpleNamespace(ads=[ad], manifest={"built_at": "2026-10-06"}, filter_ads=lambda filters: [ad])
    calls = []
    def answer(question, chunks, **kwargs):
        calls.append({"question": question, "sources": chunks, **kwargs})
        yield "Reply to " + question
    with patch("jobmatch.retrieval.Retriever", return_value=fake), \
         patch("jobmatch.generation.stream_answer", side_effect=answer):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
        next(t for t in app.text_input if t.label == "OpenAI API key").set_value("test-not-real").run()
        cv = "MSc student. Python, SQL, statistics, TensorFlow. Built a classifier."
        app.text_area(key="cv").set_value(cv).run()
        app.checkbox(key="cv_consent").check().run()
        app.selectbox(key="target_ad_id").set_value("1").run()
        question = "which of my skills are relevant for machine learning"
        for text in ["Review my CV", question, "you have my cv, what do you mean", "what about deep learning?"]:
            app.chat_input[0].set_value(text).run()
            assert not app.exception and not app.error
            assert app.session_state["cv"] == cv
            assert calls[-1]["cv"] == cv and calls[-1]["sources"] == []
            assert app.session_state["messages"][-1]["active_request"] == ("Review my CV" if text == "Review my CV" else question)
        assert calls[1]["intent"] == "cv_advice"
        assert calls[2]["intent"] == calls[3]["intent"] == "follow_up"
        assert len(app.session_state["messages"]) == 8
        # Even a question without CV keywords retains consented context.
        app.chat_input[0].set_value("Explain regularization").run()
        assert calls[-1]["cv"] == cv and calls[-1]["sources"] == []
        # Without consent the CV is never attached to a general model request.
        app.checkbox(key="cv_consent").uncheck().run()
        app.chat_input[0].set_value("What is SQL?").run()
        assert calls[-1]["cv"] == ""
        count = len(calls)
        app.chat_input[0].set_value(question).run()
        assert len(calls) == count
        assert any("not enabled" in w.value for w in app.warning)


def test_page_load_bootstraps_jobs_once_and_widget_reruns_preserve_cv(monkeypatch, tmp_path, jobs_update):
    import json
    import jobmatch.config
    monkeypatch.setattr(jobmatch.config, "INDEX_PATH", tmp_path)
    ad = {"ad_id": "1", "headline": "Software Engineer", "text": "Python required", "location": "Uppsala",
          "role": "Developer", "employer": "Example", "deadline": "", "published": "",
          "url": "https://example.com/ad"}
    manifest = {"generation": "test-page-load", "built_at": "2026-10-19T12:00:00+00:00",
                "checked_at": "2026-10-19T12:01:00+00:00"}
    def update(index):
        (index / "manifest.json").write_text(json.dumps(manifest))
        return manifest
    jobs_update.side_effect = update
    fake = SimpleNamespace(ads=[ad], manifest=manifest, filter_ads=lambda filters: [ad])
    with patch("jobmatch.retrieval.Retriever", return_value=fake):
        path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(path, default_timeout=30).run()
        assert not app.exception and app.chat_input
        assert jobs_update.call_count == 1  # Automatically created the missing database.
        assert any("Last successful update: 2026-10-19 14:01 (Europe/Stockholm)" in c.value for c in app.caption)
        app.text_area(key="cv").set_value("Python and SQL projects").run()
        app.checkbox(key="cv_consent").check().run()
        app.chat_input[0].set_value("What is today's date?").run()
        app.run()
        assert not app.exception and jobs_update.call_count == 1
        assert app.session_state["cv"] == "Python and SQL projects" and app.session_state["cv_consent"]
        assert len(app.session_state["messages"]) == 2
        # A fresh browser session checks the persisted update time again.
        fresh = AppTest.from_file(path, default_timeout=30).run()
        assert not fresh.exception and jobs_update.call_count == 2


def test_update_failure_uses_saved_jobs_and_retry_keeps_conversation(monkeypatch, tmp_path, jobs_update):
    import jobmatch.config
    monkeypatch.setattr(jobmatch.config, "INDEX_PATH", tmp_path)
    (tmp_path / "manifest.json").write_text('{"generation":"offline"}')
    ad = {"ad_id": "1", "headline": "Software Engineer", "text": "SQL required", "location": "Uppsala",
          "role": "Developer", "employer": "Example", "deadline": "", "published": "",
          "url": "https://example.com/ad"}
    fake = SimpleNamespace(ads=[ad], manifest={"built_at": "2026-10-06"}, filter_ads=lambda filters: [ad])
    jobs_update.side_effect = [OSError("offline"), None]
    with patch("jobmatch.retrieval.Retriever", return_value=fake):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
        assert not app.exception and app.chat_input
        assert any("last saved database" in w.value for w in app.warning)
        app.text_area(key="cv").set_value("SQL projects").run()
        app.checkbox(key="cv_consent").check().run()
        app.chat_input[0].set_value("What is today's date?").run()
        assert jobs_update.call_count == 1
        next(b for b in app.button if b.label == "Retry jobs update").click().run()
        assert not app.exception and not app.warning
        assert jobs_update.call_count == 2
        assert app.session_state["cv"] == "SQL projects" and app.session_state["cv_consent"]
        assert len(app.session_state["messages"]) == 2


def test_first_load_offline_shows_retry_without_a_broken_chat(monkeypatch, tmp_path, jobs_update):
    import jobmatch.config
    monkeypatch.setattr(jobmatch.config, "INDEX_PATH", tmp_path)
    jobs_update.side_effect = OSError("offline")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception and app.error and not app.chat_input
    assert any(b.label == "Retry jobs update" for b in app.button)


def test_new_page_sessions_reuse_recent_database_without_network(monkeypatch, tmp_path, jobs_update):
    import json
    from datetime import datetime, timezone
    import jobmatch.config
    monkeypatch.setattr(jobmatch.config, "INDEX_PATH", tmp_path)
    ad = {"ad_id": "1", "headline": "Software Engineer", "text": "SQL required", "location": "Uppsala",
          "role": "Developer", "employer": "Example", "deadline": "", "published": "",
          "url": "https://example.com/ad"}
    generation = tmp_path / "recent"
    (generation / "chroma").mkdir(parents=True)
    (generation / "ads.json").write_text(json.dumps([ad]))
    (generation / "chunks.json").write_text('[]')
    manifest = {"generation": "recent", "scope": "tech", "embedding_model": jobmatch.config.EMBEDDING_MODEL,
                "built_at": "2026-10-06", "checked_at": datetime.now(timezone.utc).isoformat(),
                "filter_version": TECH_FILTER_VERSION}
    manifest_text = json.dumps(manifest)
    (tmp_path / "manifest.json").write_text(manifest_text)
    # Call the real freshness check on each page, without touching project data.
    jobs_update.side_effect = lambda index: refresh_saved_jobs(index, tmp_path / "ads.jsonl")
    fake = SimpleNamespace(ads=[ad], manifest=manifest, filter_ads=lambda filters: [ad])
    with patch("jobmatch.retrieval.Retriever", return_value=fake), \
         patch("jobmatch.ingest.download_ads", side_effect=AssertionError("Recent jobs must not download")) as download, \
         patch("jobmatch.retrieval.build_index", side_effect=AssertionError("Recent jobs must not rebuild")) as build:
        for _ in range(2):
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
            assert not app.exception and not app.error and not app.warning
            assert app.chat_input and not app.session_state["jobs_refresh_failed"]
        assert jobs_update.call_count == 2
        download.assert_not_called()
        build.assert_not_called()
    assert (tmp_path / "manifest.json").read_text() == manifest_text


def test_thesis_library_and_chat_filters_do_not_confuse_past_cv_degrees_with_openings(monkeypatch, tmp_path):
    import jobmatch.config
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(jobmatch.config, "INDEX_PATH", tmp_path)
    (tmp_path / "manifest.json").write_text('{}')
    common = {"text": "Machine learning with Python", "employer": "Example", "location": "Uppsala",
              "role": "Engineer", "deadline": "", "published": "", "url": "https://example.com/ad"}
    ads = [{**common, "ad_id": "1", "headline": "Software Engineer", "opportunity_type": "job"},
           {**common, "ad_id": "2", "headline": "Master Thesis", "opportunity_type": "thesis"}]
    calls = []
    def filtered(filters):
        return [ad for ad in ads if all(not value or ad.get(key) == value for key, value in filters.items())]
    def retrieve(query, k, filters, mode, similarity):
        calls.append((query, filters))
        return [{"chunk_id": ad["ad_id"] + ":0", "text": ad["text"],
                 "metadata": {key: value for key, value in ad.items() if key != "text"}} for ad in filtered(filters)]
    fake = SimpleNamespace(ads=ads, manifest={"built_at": "2026-10-19"}, filter_ads=filtered, retrieve=retrieve)
    def answer(*args, **kwargs):
        kwargs["stats"].citation_ids = [1]
        yield "A relevant opportunity [1]."
    with patch("jobmatch.retrieval.Retriever", return_value=fake), patch("jobmatch.generation.stream_answer", side_effect=answer):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
        app.radio(key="workspace_view").set_value("Job library").run()
        assert len(app.dataframe[0].value) == 2
        next(s for s in app.selectbox if s.label == "Opportunity type").set_value("Thesis opportunities").run()
        assert list(app.dataframe[0].value["headline"]) == ["Master Thesis"]
        next(s for s in app.selectbox if s.label == "Opportunity type").set_value("All opportunities").run()
        app.radio(key="workspace_view").set_value("Chat").run()
        app.chat_input[0].set_value("Find master's thesis opportunities in machine learning").run()
        assert calls[-1][1]["opportunity_type"] == "thesis"
        assert {c["metadata"]["opportunity_type"] for c in app.session_state["messages"][-1]["sources"]} == {"thesis"}
        app.text_area(key="cv").set_value("Completed a master's thesis in machine learning. Python.").run()
        app.checkbox(key="cv_consent").check().run()
        app.chat_input[0].set_value("Find jobs matching my CV").run()
        assert "master's thesis" in calls[-1][0]
        assert calls[-1][1]["opportunity_type"] is None
        assert not app.exception
