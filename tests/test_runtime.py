"""Regression checks for clock facts and follow-ups, without model/network calls."""
import json
from dataclasses import asdict
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from jobmatch.conversation import intent_for
from jobmatch.generation import GenerationStats, stream_answer
from jobmatch.scope import IN_SCOPE
from jobmatch.runtime import runtime_context


CLOCK = runtime_context(datetime.fromisoformat("2026-10-19T14:30:00+00:00"))
JOB = {"text": "SQL required.", "metadata": {"ad_id": "1", "employer": "Nordea"}}


@pytest.mark.parametrize("question,task", [
    ("what is today's date", "clock"),
    ("so what is today’s date", "clock"),
    ("What day of the week is it?", "clock"),
    ("What's the time?", "clock"),
    ("What is the current time?", "clock"),
    ("how do you know this", "follow_up"),
    ("how", "follow_up"),
    ("Are you sure?", "follow_up"),
    ("when was the Nordea job ad listed", "job_question"),
    ("What is the job's closing date?", "job_question"),
])
def test_clock_and_followup_routing(question, task):
    assert intent_for(question) == task


@pytest.mark.parametrize("instant,date,offset", [
    ("2026-12-31T23:30:00+00:00", "2027-01-01", "+01:00"),
    ("2026-06-30T22:30:00+00:00", "2026-07-01", "+02:00"),
])
def test_stockholm_clock_handles_midnight_and_daylight_saving(instant, date, offset):
    context = runtime_context(datetime.fromisoformat(instant))
    assert context["current_date"] == date
    assert context["current_datetime"].endswith(offset)
    assert context["timezone"] == "Europe/Stockholm"


@pytest.mark.parametrize("backend", ["openai", "ollama"])
def test_screenshot_sequence_uses_clock_even_with_old_jobs_and_incorrect_history(backend):
    history = [{"role": "user", "content": "When was the Nordea job ad listed?"},
               {"role": "assistant", "content": "5 October 2026 [1].", "sources": [JOB]},
               {"role": "user", "content": "what is today's date"},
               {"role": "assistant", "content": "Today's date is June 25, 2024.", "sources": []}]
    with patch("jobmatch.generation.runtime_context", return_value=CLOCK), \
         patch("openai.OpenAI", side_effect=AssertionError("Clock must not use a model")), \
         patch("jobmatch.ollama.require_model", side_effect=AssertionError("Clock must not use Ollama")):
        for question in ["how do you know this", "so what is today's date", "how"]:
            stats = GenerationStats()
            answer = "".join(stream_answer(question, [JOB], history=history, backend=backend,
                                          model="", stats=stats))
            assert "19 October 2026" in answer and "Europe/Stockholm" in answer
            assert "2024" not in answer and "Nordea" not in answer and "[1]" not in answer
            assert stats.answer_origin == "runtime_clock" and not stats.citation_ids
            if question.startswith("how"):
                assert "system clock" in answer
            history.extend([{"role": "user", "content": question},
                            {"role": "assistant", "content": answer, "sources": [], "stats": asdict(stats)}])


def test_clock_is_read_again_for_each_question():
    tomorrow = runtime_context(datetime.fromisoformat("2026-10-20T00:01:00+02:00"))
    with patch("jobmatch.generation.runtime_context", side_effect=[CLOCK, tomorrow]):
        first = "".join(stream_answer("what is today's date", []))
        second = "".join(stream_answer("what is today's date", []))
    assert "19 October 2026" in first and "20 October 2026" in second


@pytest.mark.parametrize("immediate_sources", [[], [JOB]])
def test_followup_uses_only_immediate_answer_sources_and_runtime_context(immediate_sources):
    history = [{"role": "user", "content": "Find Nordea jobs"},
               {"role": "assistant", "content": "OLD JOB ANSWER [1]", "sources": [JOB]},
               {"role": "user", "content": "How should I structure a CV?"},
               {"role": "assistant", "content": "IMMEDIATE ANSWER", "sources": immediate_sources}]
    events = [SimpleNamespace(type="response.output_text.delta", delta=IN_SCOPE + "\nHere is the explanation ["),
              SimpleNamespace(type="response.output_text.delta", delta="1].")]
    with patch("jobmatch.generation.runtime_context", return_value=CLOCK), patch("openai.OpenAI") as constructor:
        client = constructor.return_value.__enter__.return_value
        client.responses.create.return_value.__enter__.return_value = iter(events)
        stats = GenerationStats()
        answer = "".join(stream_answer("how do you know this", [JOB], history=history, api_key="test", stats=stats))
        request = client.responses.create.call_args.kwargs
    messages = request["input"]
    payload = json.loads(messages[-1]["content"])
    assert len(payload["sources"]) == len(immediate_sources)
    assert len(messages[:-1]) == 2
    assert "OLD JOB ANSWER" not in json.dumps(messages)
    assert payload["runtime_context"] == CLOCK
    assert "2026-10-19" in request["instructions"]
    assert "immediately preceding" in request["instructions"]
    assert answer == ("Here is the explanation [1]." if immediate_sources else "Here is the explanation.")
    assert not stats.invalid_citation_ids


def test_job_publication_question_still_uses_model_and_job_sources():
    events = [SimpleNamespace(type="response.output_text.delta", delta=IN_SCOPE + "\n5 October 2026 [1].")]
    with patch("jobmatch.generation.runtime_context", return_value=CLOCK), patch("openai.OpenAI") as constructor:
        client = constructor.return_value.__enter__.return_value
        client.responses.create.return_value.__enter__.return_value = iter(events)
        answer = "".join(stream_answer("When was the Nordea job ad listed?", [JOB], api_key="test"))
    assert answer == "5 October 2026 [1]."
