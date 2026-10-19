"""UI regression checks use local fake retrieval and no external API calls."""
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from streamlit.testing.v1 import AppTest


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
        app.text_area(key="cv").set_value("SQL projects").run()
        app.chat_input[0].set_value("what jobs can i get with my experience").run()
        assert not app.session_state["messages"]
        assert any("CV is attached but not enabled" in warning.value for warning in app.warning)
        app.checkbox(key="cv_consent").check().run()
        app.selectbox(key="provider").select("Ollama (local)").run()
        assert not app.checkbox(key="cv_consent").value
        next(b for b in app.button if b.label == "Clear conversation and CV").click().run()
        assert not app.session_state["messages"] and app.session_state["cv"] == ""
        assert not app.exception
