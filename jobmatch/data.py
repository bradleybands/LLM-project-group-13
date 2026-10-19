"""Normalize original JobTech ads without translating or changing requirements."""
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

THESIS_PATTERN = re.compile(
    r"\b(?:master(?:['’]s|s)?[ -]+(?:degree[ -]+)?(?:thes[ie]s|project)|"
    r"m\.?\s?sc\.?[ -]+(?:thes[ie]s|project)|thes[ie]s|degree[ -]+project|"
    r"examensarbet\w*|exjobb\w*|master(?:uppsats|arbete)\w*)\b", re.IGNORECASE,
)


def is_thesis_ad(ad: dict) -> bool:
    # A degree/thesis mentioned in the requirements of a normal job is not
    # itself a thesis opening. Keep classification tied to the advertised role.
    occupation = ad.get("occupation") or {}
    role = " ".join([ad.get("headline") or "", ad.get("role") or "", occupation.get("label") or ""])
    return bool(THESIS_PATTERN.search(role))


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def normalize_ad(raw: dict) -> dict:
    description = raw.get("description") or {}
    if isinstance(description, dict):
        text = description.get("text") or description.get("text_formatted") or ""
    else:
        text = str(description)
    workplace = raw.get("workplace_address") or {}
    employer = raw.get("employer") or {}
    occupation = raw.get("occupation") or {}
    field = raw.get("occupation_field") or {}
    ad_id = str(raw.get("id") or "")
    return {
        "ad_id": ad_id,
        "headline": clean_text(raw.get("headline") or ""),
        "text": clean_text(text),
        "employer": employer.get("name") or "Not specified",
        "location": workplace.get("municipality") or workplace.get("city") or "Not specified",
        "region": workplace.get("region") or "Not specified",
        "role": occupation.get("label") or "Not specified",
        "field": field.get("label") or "Not specified",
        "opportunity_type": "thesis" if is_thesis_ad(raw) else "job",
        "url": raw.get("webpage_url") or f"https://arbetsformedlingen.se/platsbanken/annonser/{ad_id}",
        "published": raw.get("publication_date") or "",
        "deadline": raw.get("application_deadline") or "",
    }


def load_ads(path: Path) -> list[dict]:
    ads, seen_ids, seen_texts = [], set(), set()
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("expected an object")
                ad = normalize_ad(raw)
            except (ValueError, AttributeError, TypeError) as exc:
                raise ValueError(f"Invalid ad at {path}:{number}: {exc}") from exc
            digest = hashlib.sha256(ad["text"].encode()).hexdigest()
            if not ad["ad_id"] or not ad["text"] or ad["ad_id"] in seen_ids or digest in seen_texts:
                continue
            ads.append(ad)
            seen_ids.add(ad["ad_id"])
            seen_texts.add(digest)
    if not ads:
        raise ValueError("No non-empty, unique ads found in the input file")
    return ads


def ads_fingerprint(ads: list[dict]) -> str:
    """Compare searchable content independently of download order/unused fields."""
    content = json.dumps(sorted(ads, key=lambda ad: ad["ad_id"]), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def is_expired(ad: dict) -> bool:
    if not ad.get("deadline"):
        return False
    try:
        return datetime.fromisoformat(ad["deadline"].replace("Z", "+00:00")).date() < datetime.now(timezone.utc).date()
    except ValueError:
        return False


def make_chunks(ads: list[dict], tokenizer, size: int = 320, overlap: int = 60) -> list[dict]:
    if not 32 <= size <= 450 or not 0 <= overlap < size:
        raise ValueError("Chunk size must be 32–450 tokens; overlap must be smaller than size")
    chunks = []
    for ad in ads:
        # Include role/location in each embedding while reserving room for E5's 512-token limit.
        header = f"{ad['headline']} | {ad['employer']} | {ad['location']} | {ad['role']}\n"
        header_tokens = tokenizer.encode(header, add_special_tokens=False)[:40]
        header = tokenizer.decode(header_tokens, skip_special_tokens=True)
        # Tokenize the full description for slicing, not for a model forward pass.
        tokens = tokenizer.encode(ad["text"], add_special_tokens=False, verbose=False)
        for start in range(0, len(tokens), size - overlap):
            body = tokenizer.decode(tokens[start:start + size], skip_special_tokens=True)
            chunks.append({
                "chunk_id": f"{ad['ad_id']}:{start}",
                "text": header + "\n" + body,
                "metadata": {key: value for key, value in ad.items() if key != "text"},
            })
            if start + size >= len(tokens):
                break
    return chunks
