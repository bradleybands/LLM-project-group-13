"""Download a local snapshot of JobTech job ads for JobMatch."""

import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SNAPSHOT_URL = "https://jobstream.api.jobtechdev.se/v2/snapshot"

# Match explicit role names and methods, rather than broad words like "data"
# or "Python" which appear in many unrelated ads. Swedish and English terms.
DATA_SCIENCE_PATTERN = re.compile(
    r"\b(?:data[ -]scien(?:ce|tist)|datavetenskap\w*|"
    r"machine[ -]learning|maskininlärning\w*|deep[ -]learning|"
    r"data[ -]analyst|dataanalytiker\w*|data[ -]engineer\w*|"
    r"analytics[ -]engineer\w*|ml[ -]engineer\w*|"
    r"ai[ -]engineer\w*|business intelligence|"
    r"statistiker\w*|statistician\w*)\b",
    re.IGNORECASE,
)


def is_data_science_ad(ad: dict) -> bool:
    """Keyword heuristic: a match indicates relevance, not a guaranteed role."""
    description = ad.get("description") or {}
    text = description.get("text", "") if isinstance(description, dict) else str(description)
    occupation = ad.get("occupation") or {}
    searchable = " ".join([
        ad.get("headline") or "", occupation.get("label") or "", text or "",
    ])
    return bool(DATA_SCIENCE_PATTERN.search(searchable))


def download_ads(
    output: Path, limit: int, it_only: bool = False, data_science_only: bool = False
) -> int:
    """Save ads as JSONL, preserving the original fields for later cleaning."""
    request = Request(SNAPSHOT_URL, headers={"Accept": "application/jsonl"})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    count = 0
    try:
        with urlopen(request, timeout=120) as response, temporary.open(
            "w", encoding="utf-8"
        ) as destination:
            for line in response:
                if not line.strip():
                    continue
                ad = json.loads(line)
                if not isinstance(ad, dict):
                    raise ValueError("Expected a job-ad object in each JSONL line")
                # Filter before counting: the limit refers to matching ads saved.
                field = ad.get("occupation_field") or {}
                if it_only and field.get("label", "").casefold() != "data/it":
                    continue
                if data_science_only and not is_data_science_ad(ad):
                    continue
                destination.write(json.dumps(ad, ensure_ascii=False) + "\n")
                count += 1
                if count <= 3:
                    print(f"  {ad.get('id', '')}: {ad.get('headline', '(no headline)')}")
                if count >= limit:
                    break
        if count == 0:
            raise ValueError("JobTech returned no ads matching the selected filter")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count
