import io
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request
from unittest.mock import patch

import pytest

from jobmatch.ollama import LocalModelError, installed_models, open_local_request, require_model
from jobmatch.generation import GenerationStats, stream_answer, is_cv_match_question
from jobmatch.scope import IN_SCOPE


def test_unavailable_server_has_actionable_error():
    with patch("jobmatch.ollama.urlopen", side_effect=URLError("connection refused")):
        with pytest.raises(LocalModelError, match="run.sh --ollama"):
            installed_models()


def test_missing_model_is_detected_before_generation():
    with patch("jobmatch.ollama.installed_models", return_value=["qwen2.5:3b"]):
        with pytest.raises(LocalModelError, match="not been downloaded"):
            require_model("qwen2.5:7b")
    with patch("jobmatch.ollama.installed_models", return_value=["qwen2.5:latest"]):
        require_model("qwen2.5")


def test_http_error_never_echoes_response_contents():
    exc = HTTPError("http://localhost", 404, "secret request contents", {}, io.BytesIO(b"private CV"))
    with patch("jobmatch.ollama.urlopen", side_effect=exc):
        with pytest.raises(LocalModelError) as caught:
            open_local_request(Request("http://localhost"))
    assert "secret" not in str(caught.value) and "private" not in str(caught.value)
    assert "ollama pull" in str(caught.value)


def test_local_streaming_and_usage():
    events = [{"message": {"content": IN_SCOPE + "\nSQL required [1]."}, "done": False},
              {"message": {"content": ""}, "done": True, "prompt_eval_count": 100, "eval_count": 10}]
    response = io.BytesIO(b"\n".join(json.dumps(event).encode() for event in events))
    with patch("jobmatch.ollama.require_model"), patch("jobmatch.ollama.open_local_request", return_value=response):
        stats = GenerationStats()
        answer = "".join(stream_answer("Skills?", [{"text": "SQL required", "metadata": {"ad_id": "1"}}],
                                      backend="ollama", model="qwen2.5:7b", stats=stats))
    assert answer == "SQL required [1]." and stats.citation_ids == [1]
    assert stats.input_tokens == 100 and stats.output_tokens == 10


def test_natural_cv_queries_use_evidence_comparison():
    assert is_cv_match_question("what jobs can i get with my experience")
    assert is_cv_match_question("Which jobs fit this CV?")
    assert not is_cv_match_question("Rewrite my CV bullet for this role")
    assert not is_cv_match_question("What benefits does this role offer?")
