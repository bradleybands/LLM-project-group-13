#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "${1:-}" = "--ollama" ]; then
  export OLLAMA_MODELS="$PWD/.cache/ollama"
  export OLLAMA_HOST="127.0.0.1:11434"
  if command -v ollama >/dev/null 2>&1; then
    exec ollama serve
  elif [ -x /opt/homebrew/bin/ollama ]; then
    exec /opt/homebrew/bin/ollama serve
  fi
  echo "Install Ollama first: https://ollama.com/download"
  exit 1
fi
if [ ! -x .venv/bin/streamlit ]; then
  echo "Install the virtual environment and requirements first; see README.md."
  exit 1
fi
exec .venv/bin/streamlit run app.py "$@"
