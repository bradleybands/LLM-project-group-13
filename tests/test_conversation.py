import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from jobmatch.conversation import intent_for, resolve_job, focused_sources, uses_cv, active_request, thesis_opportunity_query
from jobmatch.generation import stream_answer, REFUSAL
from jobmatch.scope import IN_SCOPE


@pytest.mark.parametrize("question,expected", [
    ("How can I tailor my cv for a specific job?", "tailor"),
    ("How can I tailor my v to the Nextory job?", "tailor"),
    ("Rewrite my second bullet for this role", "rewrite"),
    ("Make it shorter", "rewrite"),
    ("Review my CV", "cv_advice"),
    ("what jobs can i get with my experience", "cv_match"),
    ("Which roles fit my CV?", "cv_match"),
    ("Find machine learning jobs in Stockholm", "job_search"),
    ("Find master's thesis opportunities in machine learning", "job_search"),
    ("master thesis opportunities", "job_search"),
    ("Which master's thesis fits my CV?", "cv_match"),
    ("Show me examensarbete opportunities", "job_search"),
    ("What language requirements does this role state?", "job_question"),
    ("How do I structure a CV?", "advice"),
    ("What is SQL?", "advice"),
    ("What does pattern matching mean?", "advice"),
    ("Compare Python and SQL", "advice"),
    ("Compare these two jobs", "job_question"),
    ("How can I improve my skills?", "cv_advice"),
    ("which of my skills are relevant for machine learning", "cv_advice"),
    ("Which of my skills fit machine learning?", "cv_advice"),
    ("Hello", "chat"),
    ("Thanks", "chat"),
])
def test_question_intent(question, expected):
    assert intent_for(question) == expected


@pytest.mark.parametrize("question,expected", [
    ("Find master's thesis opportunities in machine learning", True),
    ("Which theses require Python?", True),
    ("Find jobs requiring a completed master's thesis", False),
    ("How do I write a master's thesis?", False),
    ("Which skills from my completed thesis should I put on my CV?", False),
])
def test_only_thesis_opening_queries_narrow_the_job_type(question, expected):
    assert thesis_opportunity_query(question) is expected


ADS = [
    {"ad_id": "11111111", "headline": "Junior Data Engineer", "employer": "Nextory AB", "location": "Stockholm", "text": "Some SQL and basic Python preferred."},
    {"ad_id": "22222222", "headline": "Data Scientist", "employer": "Embark Studios AB", "location": "Stockholm", "text": "Strong SQL and professional English required."},
]


def test_job_references_and_ambiguous_followup():
    sources = [
        {"metadata": {"ad_id": "11111111"}},
        {"metadata": {"ad_id": "11111111"}},
        {"metadata": {"ad_id": "22222222"}},
    ]
    history = [{"role": "assistant", "sources": sources, "content": "Example jobs"}]
    assert resolve_job("Tailor it for the second job", ADS, history) == "22222222"
    assert resolve_job("Tailor it for [2]", ADS, history) == "11111111"
    assert resolve_job("Tailor it for Nextory", ADS, history) == "11111111"
    assert resolve_job("Tailor it for job 22222222", ADS, history) == "22222222"
    assert resolve_job("Tailor it for this job", ADS, history) is None
    assert resolve_job("Tailor it for this job", ADS, history, "22222222") == "22222222"
    assert resolve_job("Tailor it for [99]", ADS, history, "22222222") is None


def test_focused_job_excludes_unrelated_jobs():
    sources = focused_sources(ADS, "11111111")
    assert len(sources) == 1 and sources[0]["metadata"]["employer"] == "Nextory AB"
    assert "professional English" not in sources[0]["text"]
    assert uses_cv("Tailor my CV for this role")
    assert uses_cv("Rewrite my summary")
    assert not uses_cv("Hello")


def test_short_job_selection_continues_tailoring_task():
    history = [{"role": "user", "content": "How do I tailor my CV for a specific job?"},
               {"role": "assistant", "content": "Which job would you like to target?"}]
    assert intent_for("the second one", history) == "tailor"
    assert intent_for("[2]", history) == "tailor"
    assert intent_for("What is SQL?", history) == "advice"
    assert intent_for("How can I improve my CV for this role?") == "tailor"


@pytest.mark.parametrize("question,cv,chunks", [
    ("How do I structure a CV?", "", []),
    ("How can I tailor my CV for a specific job?", "MSc student. SQL. Built a classifier.", []),
    ("Hello", "", []),
])
def test_advice_and_tailoring_are_not_forced_into_job_lists_or_json(question, cv, chunks):
    events = [SimpleNamespace(type="response.output_text.delta", delta=IN_SCOPE + "\nUse a clear, truthful project bullet."),
              SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=None))]
    with patch("openai.OpenAI") as constructor:
        client = constructor.return_value.__enter__.return_value
        client.responses.create.return_value.__enter__.return_value = iter(events)
        answer = "".join(stream_answer(question, chunks, cv=cv, api_key="test"))
        options = client.responses.create.call_args.kwargs
        assert answer != REFUSAL and "text" not in options
        assert "CURRENT TASK:" in options["instructions"]
        assert json.loads(options["input"][-1]["content"])["question"] == question


def test_tailoring_displays_edits_and_checked_copy_not_job_catalogue():
    from jobmatch.writing import render_writing, supported_rewrite
    chunks = focused_sources(ADS, "11111111")
    payload = {"items": [{"source": 1, "source_quote": "Some SQL and basic Python preferred.",
                           "cv_quote": "Built a classifier using Python.", "action": "prioritize"}],
               "draft_original": "Built a classifier using Python.",
               "draft_text": "Developed a classifier using Python."}
    answer = render_writing(payload, chunks, "Built a classifier using Python.")
    assert "Move this evidence higher" in answer and "Suggested wording" in answer
    assert "Developed a classifier using Python." in answer
    assert "Original evidence:" not in answer and "Embark" not in answer
    assert not supported_rewrite("Python and SQL skills.", "Proficient in Airbyte and advanced SQL.")
    assert not supported_rewrite("MSc student with basic SQL.", "MSc with SQL.")
    assert not supported_rewrite("Built a classifier.", "Built a classifier with 95% accuracy.")
    payload["draft_text"] = "Experienced data scientist proficient in Airbyte."
    answer = render_writing(payload, chunks, "Built a classifier using Python.")
    assert "proficient in Airbyte" not in answer and "Suggested wording" in answer


def test_rewrite_uses_structured_validation_but_renders_plain_cv_text():
    cv = "Built a classifier using Python."
    payload = {"in_scope": True, "items": [], "draft_original": cv, "draft_text": "Developed a classifier using Python."}
    events = [SimpleNamespace(type="response.output_text.delta", delta=json.dumps(payload)),
              SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=None))]
    with patch("openai.OpenAI") as constructor:
        client = constructor.return_value.__enter__.return_value
        client.responses.create.return_value.__enter__.return_value = iter(events)
        answer = "".join(stream_answer("Rewrite my project bullet", [], cv=cv, api_key="test"))
        assert "Suggested wording" in answer and "Developed a classifier" in answer
        assert '"items"' not in answer


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-4.1-mini"])
def test_model_upgrade_preserves_streaming_and_structured_cv_contract(model):
    from jobmatch.generation import GenerationStats
    cv = "Built a classifier using Python."
    payload = {"in_scope": True, "items": [], "draft_original": cv, "draft_text": "Developed a classifier using Python."}
    usage = SimpleNamespace(input_tokens=100, output_tokens=600,
                            output_tokens_details=SimpleNamespace(reasoning_tokens=400))
    events = [SimpleNamespace(type="response.output_text.delta", delta=json.dumps(payload)),
              SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=usage))]
    with patch("openai.OpenAI") as constructor:
        client = constructor.return_value.__enter__.return_value
        client.responses.create.return_value.__enter__.return_value = iter(events)
        stats = GenerationStats()
        answer = "".join(stream_answer("Rewrite my project bullet", [], cv=cv, model=model, api_key="test", stats=stats))
        request = client.responses.create.call_args.kwargs
    assert "Suggested wording" in answer and "Developed a classifier" in answer
    assert request["text"]["format"]["strict"] is True
    assert request["stream"] is True and request["store"] is False
    if model == "gpt-6-luna":
        assert request["reasoning"] == {"effort": "low"}
        assert request["max_output_tokens"] >= 8000
        assert stats.reasoning_tokens == 400
    else:
        # Older non-reasoning models must not receive unsupported parameters.
        assert "reasoning" not in request
        assert request["max_output_tokens"] == 3500


def test_project_edit_does_not_rewrite_the_entire_cv_or_add_traits():
    from jobmatch.writing import render_writing
    cv = "MSc student in Data Science. Skills: Python and SQL. Project: Built a classification model using Python and scikit-learn."
    payload = {"items": [], "draft_original": cv,
               "draft_text": "Experienced, structured data scientist proficient in Airbyte."}
    answer = render_writing(payload, focused_sources(ADS, "11111111"), cv, "rewrite", "Make the project bullet shorter")
    assert "Built a classifier with Python and scikit-learn." in answer
    assert "MSc student" not in answer
    assert "Airbyte" not in answer and "proficient" not in answer
    payload["draft_text"] = "Developed a classification model using Python and scikit-learn."
    answer = render_writing(payload, focused_sources(ADS, "11111111"), cv, "rewrite", "Make the project bullet shorter")
    assert "Built a classifier with Python and scikit-learn." in answer


def test_long_history_preserves_the_draft_at_the_end():
    from jobmatch.generation import build_input
    history = [{"role": "assistant", "content": "Background " * 1000 + "\nDRAFT: Built a classifier using Python."}]
    payload = json.loads(build_input("Make it shorter", [], history=history))
    assert "DRAFT: Built a classifier using Python." in payload["previous_conversation"][0]["content"]
    assert len(payload["previous_conversation"][0]["content"]) <= 3000


def test_cv_correction_keeps_original_request_through_multiple_followups():
    question = "which of my skills are relevant for machine learning"
    history = [{"role": "user", "content": "Review my CV"},
               {"role": "assistant", "content": "Here is a full CV review.", "active_request": "Review my CV"},
               {"role": "user", "content": question},
               {"role": "assistant", "content": "No CV was provided.", "active_request": question}]
    for followup in ["you have my cv, what do you mean", "what about deep learning?", "Is that relevant?", "why?"]:
        assert intent_for(followup, history) == "follow_up"
        assert active_request(followup, history) == question
        assert uses_cv(followup, history)
        history.extend([{"role": "user", "content": followup},
                        {"role": "assistant", "content": "Skills explanation", "active_request": question}])
    assert active_request("What is today's date?", history) == "What is today's date?"
    assert not uses_cv("What is today's date?", history)


@pytest.mark.parametrize("backend", ["openai", "ollama"])
def test_native_history_and_full_cv_survive_correction_on_both_backends(backend):
    import io
    cv = "MSc student. Skills: Python, SQL, statistics, TensorFlow and scikit-learn. Built a classifier."
    question = "Which of my skills are relevant for machine learning?"
    history = [{"role": "user", "content": "Review my CV"},
               {"role": "assistant", "content": "Old CV review", "sources": []},
               {"role": "user", "content": question},
               {"role": "assistant", "content": "No CV was supplied.", "sources": [], "active_request": question}]
    correction = "you have my cv, what do you mean"
    if backend == "openai":
        with patch("openai.OpenAI") as constructor:
            client = constructor.return_value.__enter__.return_value
            events = [SimpleNamespace(type="response.output_text.delta", delta=IN_SCOPE + "\nPython and statistics are relevant.")]
            client.responses.create.return_value.__enter__.return_value = iter(events)
            list(stream_answer(correction, [], cv=cv, history=history, api_key="test"))
            request = client.responses.create.call_args.kwargs
            messages, instructions = request["input"], request["instructions"]
            assert request["store"] is False
    else:
        response = io.BytesIO(json.dumps({"message": {"content": IN_SCOPE + "\nPython and statistics are relevant."}, "done": True}).encode())
        with patch("jobmatch.ollama.require_model"), patch("jobmatch.ollama.open_local_request", return_value=response) as send:
            list(stream_answer(correction, [], cv=cv, history=history, backend="ollama", model="qwen2.5:7b"))
            request = json.loads(send.call_args.args[0].data)
            instructions = request["messages"][0]["content"]
            messages = request["messages"][1:]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == question
    assert messages[1]["content"] == "No CV was supplied."
    payload = json.loads(messages[-1]["content"])
    assert payload["question"] == correction and payload["cv"] == cv
    assert payload["conversation_context"] == {"task": "follow_up", "active_request": question, "cv_available": True}
    assert "Old CV review" not in json.dumps(messages)
    assert "do not ask to upload it again" in instructions


def test_active_request_is_retained_even_after_history_budget_is_exceeded():
    from jobmatch.generation import build_input
    question = "Which of my skills are relevant for machine learning?"
    history = [{"role": "user", "content": question}]
    for _ in range(10):
        history.extend([{"role": "assistant", "content": "Long explanation " * 500, "active_request": question},
                        {"role": "user", "content": "what do you mean?"}])
    history.append({"role": "assistant", "content": "Final explanation", "active_request": question})
    cv = "Python and statistics."
    payload = json.loads(build_input("Tell me more", [], cv=cv, history=history))
    assert payload["conversation_context"]["active_request"] == question
    assert payload["cv"] == cv
    assert sum(len(m["content"]) for m in payload["previous_conversation"]) <= 8000


def test_nearly_full_history_budget_does_not_reintroduce_an_entire_old_message():
    from jobmatch.generation import build_input
    history = [{"role": "assistant", "content": "OLD " * 2000},
               {"role": "assistant", "content": "a" * 1999},
               {"role": "assistant", "content": "b" * 3000},
               {"role": "assistant", "content": "c" * 3000}]
    payload = json.loads(build_input("Explain regularization", [], history=history))
    assert sum(len(m["content"]) for m in payload["previous_conversation"]) == 8000
