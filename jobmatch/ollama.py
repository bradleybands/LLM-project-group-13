"""Local model availability and safe, actionable backend errors."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LocalModelError(RuntimeError):
    """Messages are safe to display and never include request/CV contents."""


def ollama_url() -> str:
    return os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")


def open_local_request(request: Request, timeout: int = 180):
    try:
        return urlopen(request, timeout=timeout)
    except HTTPError as exc:
        if exc.code == 404:
            raise LocalModelError("The selected model is not available in Ollama. Download it with 'ollama pull <model-name>', then retry.") from exc
        raise LocalModelError(f"Ollama rejected the request (HTTP {exc.code}). Check the selected model and Ollama version.") from exc
    except (URLError, OSError) as exc:
        raise LocalModelError("Cannot connect to Ollama. Start it with './run.sh --ollama' (or open the Ollama app), then retry. OpenAI is also available in Model settings.") from exc


def installed_models() -> list[str]:
    request = Request(ollama_url() + "/api/tags")
    try:
        with open_local_request(request, timeout=3) as response:
            data = json.load(response)
        return sorted({model["name"] for model in data.get("models", [])})
    except (ValueError, KeyError, TypeError) as exc:
        raise LocalModelError("The Ollama endpoint returned an unexpected response. Check OLLAMA_URL.") from exc


def require_model(model: str):
    names = installed_models()
    requested = model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"
    if requested not in names and model not in names:
        raise LocalModelError("The selected model has not been downloaded. Use the exact model name shown in Model settings, or run 'ollama pull <model-name>'.")
