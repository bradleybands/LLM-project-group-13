"""Scope decisions must precede displayed text on every generation path."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from jobmatch.generation import GenerationStats, stream_answer
from jobmatch.scope import IN_SCOPE, OUT_OF_SCOPE, SCOPE_REFUSAL, ScopeStream


JOB = {"chunk_id": "1:0", "text": "Python required.", "metadata": {
    "ad_id": "1", "headline": "Data Analyst", "employer": "Example AB",
    "location": "Stockholm", "role": "Analyst", "published": "", "deadline": "",
    "url": "https://example.com/job"}}
CV = "MSc student. Skills: Python and SQL."
HISTORY = [{"role": "user", "content": "Which jobs fit my CV?"},
           {"role": "assistant", "content": "This job requires Python [1].", "sources": [JOB],
            "active_request": "Which jobs fit my CV?"}]


def test_scope_header_can_be_split_at_every_character_without_leaking():
    raw = IN_SCOPE + "\nPython is relevant for data analysis."
    stream = ScopeStream()
    output = []
    for character in raw:
        output.append(stream.feed(character))
    assert "".join(output) == "Python is relevant for data analysis."
    assert not any(output[:len(IN_SCOPE) + 1])
    assert stream.allowed is True


@pytest.mark.parametrize("raw", [OUT_OF_SCOPE + "\nUnrelated answer that must never appear.",
                                     "Unclassified answer\nMore unrelated text", "```\n" + IN_SCOPE])
def test_rejected_or_malformed_header_releases_no_answer(raw):
    stream = ScopeStream()
    assert "".join(stream.feed(character) for character in raw) == ""
    assert stream.allowed is False


@pytest.mark.parametrize("backend", ["openai", "ollama"])
@pytest.mark.parametrize("raw", [OUT_OF_SCOPE, OUT_OF_SCOPE + "\nMessi is better [1].",
                                    "Messi is better [1].", "JOBMATCH_IN_"])
def test_off_topic_text_cannot_leak_despite_cv_job_and_history(backend, raw):
    stats = GenerationStats()
    if backend == "openai":
        with patch("openai.OpenAI") as constructor:
            client = constructor.return_value.__enter__.return_value
            client.responses.create.return_value.__enter__.return_value = iter(
                SimpleNamespace(type="response.output_text.delta", delta=char) for char in raw)
            answer = "".join(stream_answer("Who is better, Messi or Ronaldo?", [JOB], cv=CV,
                                          history=HISTORY, stats=stats, api_key="test"))
            assert client.responses.create.call_count == 1
            instructions = client.responses.create.call_args.kwargs["instructions"]
    else:
        events = [{"message": {"content": char}, "done": False} for char in raw]
        response = io.BytesIO(b"\n".join(json.dumps(event).encode() for event in events))
        with patch("jobmatch.ollama.require_model"), patch("jobmatch.ollama.open_local_request", return_value=response) as send:
            answer = "".join(stream_answer("Who is better, Messi or Ronaldo?", [JOB], cv=CV,
                                          history=HISTORY, stats=stats, backend=backend, model="qwen2.5:7b"))
            assert send.call_count == 1
            instructions = json.loads(send.call_args.args[0].data)["messages"][0]["content"]
    assert answer == SCOPE_REFUSAL
    assert stats.answer_origin == "scope_guard"
    assert not stats.citation_ids and not stats.invalid_citation_ids
    assert "TOPIC BOUNDARY" in instructions and "explicit new subject" in instructions


@pytest.mark.parametrize("task", ["rewrite", "tailor", "cv_match"])
@pytest.mark.parametrize("decision", [False, "true", None])
def test_structured_tasks_cannot_bypass_scope(task, decision):
    payload = {"items": [], "draft_original": CV, "draft_text": "Unrelated recipe."}
    if decision is not None:
        payload["in_scope"] = decision
    with patch("openai.OpenAI") as constructor:
        client = constructor.return_value.__enter__.return_value
        client.responses.create.return_value.__enter__.return_value = iter([
            SimpleNamespace(type="response.output_text.delta", delta=json.dumps(payload))])
        stats = GenerationStats()
        answer = "".join(stream_answer("Write a recipe", [JOB], cv=CV, api_key="test", stats=stats, intent=task))
        options = client.responses.create.call_args.kwargs
        schema = options["text"]["format"]["schema"]
        assert schema["properties"]["in_scope"] == {"type": "boolean"}
        assert "in_scope" in schema["required"]
        assert "TOPIC BOUNDARY" in options["instructions"]
    assert answer == SCOPE_REFUSAL and stats.answer_origin == "scope_guard"
    assert "recipe" not in answer


def test_in_scope_answer_keeps_streaming_without_a_second_model_call():
    with patch("openai.OpenAI") as constructor:
        client = constructor.return_value.__enter__.return_value
        client.responses.create.return_value.__enter__.return_value = iter([
            SimpleNamespace(type="response.output_text.delta", delta=IN_SCOPE[:5]),
            SimpleNamespace(type="response.output_text.delta", delta=IN_SCOPE[5:] + "\nSQL "),
            SimpleNamespace(type="response.output_text.delta", delta="is a query language.")])
        pieces = list(stream_answer("What is SQL?", [], api_key="test"))
        assert pieces == ["SQL ", "is a query language."]
        assert client.responses.create.call_count == 1


def test_ui_refusal_drops_irrelevant_sources_and_followup_context(monkeypatch, tmp_path):
    import jobmatch.config
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(jobmatch.config, "INDEX_PATH", tmp_path)
    (tmp_path / "manifest.json").write_text('{}')
    ad = {**JOB["metadata"], "text": JOB["text"]}
    fake = SimpleNamespace(ads=[ad], manifest={"built_at": "2026-10-19"}, filter_ads=lambda filters: [ad],
                           retrieve=lambda *args, **kwargs: [JOB])
    with patch("jobmatch.ingest.refresh_jobs"), patch("jobmatch.retrieval.Retriever", return_value=fake), \
         patch("openai.OpenAI") as constructor:
        client = constructor.return_value.__enter__.return_value
        def response(**options):
            from unittest.mock import MagicMock
            result = MagicMock()
            text = json.dumps({"in_scope": False, "items": [], "draft_original": "", "draft_text": ""}) if "text" in options else OUT_OF_SCOPE
            result.__enter__.return_value = iter([SimpleNamespace(type="response.output_text.delta", delta=text)])
            return result
        client.responses.create.side_effect = response
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
        app.text_area(key="cv").set_value(CV).run()
        app.checkbox(key="cv_consent").check().run()
        app.selectbox(key="target_ad_id").set_value("1").run()
        app.session_state["messages"] = list(HISTORY)
        # The generic 'what about' route would previously reuse old job sources.
        for question in ["what about Messi vs Ronaldo?", "Why?", "Rewrite a recipe for me"]:
            app.chat_input[0].set_value(question).run()
            assert not app.exception and not app.error
            result = app.session_state["messages"][-1]
            assert result["content"] == SCOPE_REFUSAL
            assert result["sources"] == [] and result["intent"] == "out_of_scope"
            assert result["active_request"] == question
            assert result["stats"]["answer_origin"] == "scope_guard"
            assert not app.warning
