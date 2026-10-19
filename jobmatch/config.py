import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
# Keep downloaded models and caches inside this project.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
INDEX_PATH = ROOT / os.getenv("JOBMATCH_INDEX", "storage/index")
ADS_PATH = ROOT / "data" / "tech_jobs.jsonl"

