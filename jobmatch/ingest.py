"""Download current JobTech ads and safely refresh the searchable database."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from filelock import FileLock

from jobmatch.config import ADS_PATH, EMBEDDING_MODEL
from jobmatch.data import ads_fingerprint, load_ads, is_thesis_ad


SNAPSHOT_URL = "https://jobstream.api.jobtechdev.se/v2/snapshot"
REFRESH_INTERVAL = timedelta(hours=6)
TECH_FILTER_VERSION = 2

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

TECH_ROLE_PATTERN = re.compile(
    r"\b(?:ai|ml|artificial intelligence|generative ai|computer science|"
    r"software|firmware|hardware|embedded|computer|electronics?|electrical|"
    r"devops|devsecops|sre|full[ -]?stack\w*|front[ -]?end\w*|back[ -]?end\w*|"
    r"cloud|cyber\w*|network\w*|database\w*|telecom\w*|robot\w*|automation\w*|"
    r"systems?[ -]+(?:design|engineer\w*|analyst\w*|administrat\w*)|ux|ui|"
    r"(?:system\w*|solution\w*|enterprise|integration|security|infrastructure|data|it)[ -]+architect\w*|"
    r"(?:python|java|javascript|typescript|c\+\+|c#|\.net|web|mobile)[ -]+develop\w*|"
    r"system(?:utveckl\w*|arkitekt\w*|design\w*|administrat\w*|ingenjör\w*|teknik\w*|specialist\w*)|"
    r"mjukvar\w*|programvar\w*|programmer\w*|nätverk\w*|informationssäker\w*|"
    r"datavetenskap\w*|datateknik\w*|dator\w*|databas\w*|dataarkitekt\w*|"
    r"elektronik\w*|telekom\w*|signalbehandling\w*|"
    r"it[- ](?:teknik\w*|säkerhet\w*|arkitekt\w*|specialist\w*|konsult\w*))\b",
    re.IGNORECASE,
)

# Body matching needs more specific phrases than title matching: "software"
# and "professional network" occur in otherwise unrelated thesis adverts.
THESIS_TECH_PATTERN = re.compile(
    r"\b(?:ai|ml|artificial intelligence|federated learning|reinforcement learning|"
    r"computer[ -](?:science|engineering|vision|networks?)|"
    r"software[ -](?:engineering|development|testing|architecture)|"
    r"embedded|firmware|programming|cyber\w*|robot\w*|automation\w*|"
    r"electronics?|electrical|signal processing|natural language processing|"
    r"neural[ -]networks?|network[ -](?:security|protocols?|architecture)|"
    r"cloud[ -](?:computing|native|infrastructure)|database\w*|"
    r"datavetenskap\w*|datateknik\w*|dator\w*|mjukvar\w*|programvar\w*|"
    r"programmer\w*|informationssäker\w*|elektronik\w*|signalbehandling\w*)\b",
    re.IGNORECASE,
)


def is_tech_ad(ad: dict) -> bool:
    field = ad.get("occupation_field") or {}
    if (field.get("label") or "").casefold() == "data/it":
        return True
    # Use the actual role, not a passing mention in a non-technical ad's body.
    occupation = ad.get("occupation") or {}
    role = " ".join([ad.get("headline") or "", occupation.get("label") or ""])
    if DATA_SCIENCE_PATTERN.search(role) or TECH_ROLE_PATTERN.search(role):
        return True
    # Thesis titles are often generic; the technical project is described in
    # the body. This exception applies only to advertised thesis opportunities.
    if is_thesis_ad(ad):
        description = ad.get("description") or {}
        if isinstance(description, dict):
            text = description.get("text") or description.get("text_formatted") or ""
        else:
            text = str(description)
        return bool(DATA_SCIENCE_PATTERN.search(text) or THESIS_TECH_PATTERN.search(text))
    return False


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
    output: Path, limit: int | None, it_only: bool = False, data_science_only: bool = False,
    tech_only: bool = False,
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
                if it_only and (field.get("label") or "").casefold() != "data/it":
                    continue
                if data_science_only and not is_data_science_ad(ad):
                    continue
                if tech_only and not is_tech_ad(ad):
                    continue
                destination.write(json.dumps(ad, ensure_ascii=False) + "\n")
                count += 1
                if count <= 3:
                    print(f"  {ad.get('id', '')}: {ad.get('headline', '(no headline)')}")
                if limit is not None and count >= limit:
                    break
        if count == 0:
            raise ValueError("JobTech returned no ads matching the selected filter")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def last_job_update(manifest: dict) -> datetime | None:
    """Read the last successful sync, independently of index rebuild time."""
    value = manifest.get("checked_at")
    if not isinstance(value, str):
        return None
    try:
        checked = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return checked.astimezone(timezone.utc) if checked.tzinfo is not None else None


def refresh_jobs(index: Path, output: Path = ADS_PATH) -> dict:
    """Reuse a successful sync for six hours; publish only complete replacements."""
    from jobmatch.retrieval import build_index, write_manifest

    index.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Browser sessions/processes must not rebuild the same database concurrently.
    with FileLock(str(index / ".refresh.lock"), timeout=300):
        manifest_path = index / "manifest.json"
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        generation = index / previous.get("generation", "missing")
        usable = ((generation / "chroma").is_dir() and (generation / "chunks.json").is_file()
                  and (generation / "ads.json").is_file())
        checked = last_job_update(previous)
        age = datetime.now(timezone.utc) - checked if checked is not None else None
        if (usable and previous.get("scope") == "tech" and previous.get("embedding_model") == EMBEDDING_MODEL
                and previous.get("filter_version") == TECH_FILTER_VERSION
                and age is not None and timedelta(0) <= age < REFRESH_INTERVAL):
            # Recheck under the shared lock so another page's successful update
            # is immediately reused, including across browser/app restarts.
            return previous
        with TemporaryDirectory(prefix=".jobmatch-refresh-", dir=output.parent) as temporary:
            source = Path(temporary) / "ads.jsonl"
            # No cap: keep all matching tech ads in the current JobTech snapshot.
            download_ads(source, None, tech_only=True)
            ads = load_ads(source)
            fingerprint = ads_fingerprint(ads)
            old_fingerprint = previous.get("ads_fingerprint")
            if not old_fingerprint and (generation / "ads.json").exists():
                old_fingerprint = ads_fingerprint(json.loads((generation / "ads.json").read_text(encoding="utf-8")))
            unchanged = (old_fingerprint == fingerprint and previous.get("embedding_model") == EMBEDDING_MODEL and usable)
            manifest = previous.copy() if unchanged else build_index(source, index, publish=False)
            manifest.update({"source": str(output.resolve()),
                             "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                             "ads_fingerprint": fingerprint, "scope": "tech",
                             "filter_version": TECH_FILTER_VERSION,
                             "checked_at": datetime.now(timezone.utc).isoformat()})
            source.replace(output)
            write_manifest(index, manifest)
            return manifest
