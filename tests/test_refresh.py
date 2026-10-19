"""Public-data updates are atomic and reflect added, changed and removed jobs."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from jobmatch.data import ads_fingerprint, load_ads, normalize_ad
from jobmatch.ingest import download_ads, is_tech_ad, refresh_jobs, TECH_FILTER_VERSION
from jobmatch.retrieval import Retriever


def raw_ad(ad_id, headline="Software Engineer", text="Build software using Python.", **extra):
    return {"id": ad_id, "headline": headline, "description": {"text": text}, **extra}


@pytest.mark.parametrize("title", [
    "AI Scientist", "Machine Learning Researcher", "ML Engineer", "Data Scientist", "Data Engineer",
    "Data Analyst", "Software Developer", "Computer Engineer", "Electrical Engineer",
    "System Design Architect", "Solutions Architect", "Systemarkitekt", "Cloud Engineer",
    "DevOps Specialist", "Nätverkstekniker", "Embedded Developer", "Java Developer",
    "IT-säkerhetsspecialist", "Fullstackutvecklare", "Robotics Researcher", "Systems Engineer",
    "Computer Architecture Researcher", "UX Designer",
])
def test_tech_roles_are_included(title):
    assert is_tech_ad(raw_ad("1", title))


@pytest.mark.parametrize("title", ["Nurse", "HR Coordinator", "Accountant", "Business Developer", "Building Architect", "Chef",
                                   "Mechanical Engineer", "Civil Engineer", "Design Engineer Pump Product Engineering"])
def test_unrelated_roles_do_not_qualify_by_body_keywords(title):
    assert not is_tech_ad(raw_ad("1", title, "Our company uses Python, AI and machine learning software."))


def test_it_category_and_technical_occupation_are_included():
    assert is_tech_ad(raw_ad("1", "Specialist", occupation_field={"label": "Data/IT"}))
    assert is_tech_ad(raw_ad("2", "Join our team", occupation={"label": "Systemutvecklare"}))
    assert not is_tech_ad(raw_ad("3", "Nurse", occupation_field={"label": None}))


@pytest.mark.parametrize("title", ["Master Thesis", "Master's Thesis", "Master’s thesis project",
                                   "Masters thesis opportunity", "MSc thesis", "M.Sc. Thesis",
                                   "Degree project", "Examensarbete 30 hp", "Exjobb", "Masteruppsats"])
def test_generic_thesis_titles_use_the_technical_project_description(title):
    ad = raw_ad("1", title, "Develop machine learning models for embedded systems.")
    assert is_tech_ad(ad)
    assert normalize_ad(ad)["opportunity_type"] == "thesis"


def test_thesis_exception_does_not_admit_unrelated_roles_or_non_tech_theses():
    assert not is_tech_ad(raw_ad("1", "Master Thesis in Marketing", "Study customer loyalty and brand perception."))
    ad = raw_ad("2", "HR Coordinator", "A master's thesis in machine learning is useful. We use software.")
    assert not is_tech_ad(ad)
    assert normalize_ad(ad)["opportunity_type"] == "job"
    assert normalize_ad(raw_ad("3", "Data Scientist", "A completed master's thesis is required."))["opportunity_type"] == "job"
    assert is_tech_ad({"headline": "Examensarbete", "description": {"text_formatted": "<p>Software development</p>"}})
    assert is_tech_ad({"headline": "Master's Thesis", "description": "Research in computer science."})
    assert not is_tech_ad(raw_ad("4", "Master Thesis in Metallurgy", "Build a professional network within the steel industry."))
    assert not is_tech_ad(raw_ad("5", "Master Thesis - Structural Analysis", "Use commercial software for finite element analysis."))


def test_page_download_has_no_arbitrary_ad_limit(tmp_path):
    records = [raw_ad("1"), raw_ad("2", "Nurse"), raw_ad("3", "AI Scientist"), raw_ad("4", "Systemarkitekt"),
               raw_ad("5", "Master Thesis", "Research machine learning for industrial control.")]
    class Response:
        def __enter__(self): return iter(json.dumps(ad).encode() for ad in records)
        def __exit__(self, *args): pass
    output = tmp_path / "ads.jsonl"
    with patch("jobmatch.ingest.urlopen", return_value=Response()):
        assert download_ads(output, None, tech_only=True) == 4
    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == ["1", "3", "4", "5"]


class Tokenizer:
    def encode(self, text, **kwargs): return text.split()
    def decode(self, tokens, **kwargs): return " ".join(tokens)


@pytest.fixture
def public_data(monkeypatch, tmp_path):
    clock = SimpleNamespace(now=datetime(2026, 10, 19, 12, tzinfo=timezone.utc))
    class TestDateTime(datetime):
        @classmethod
        def now(cls, tz=None): return clock.now.astimezone(tz)
    monkeypatch.setattr("jobmatch.ingest.datetime", TestDateTime)
    records = [raw_ad("1"), raw_ad("2", text="Build a SQL data pipeline."), raw_ad("3", text="Maintain networks.")]
    encoded = []
    def encode(texts, **kwargs):
        encoded.extend(texts)
        return np.array([[1., float(len(text))] for text in texts])
    monkeypatch.setattr("jobmatch.retrieval.embedding_model", lambda *args: SimpleNamespace(tokenizer=Tokenizer(), encode=encode))
    def download(output, limit, **kwargs):
        assert kwargs["tech_only"] and limit is None
        output.write_text("\n".join(json.dumps(ad) for ad in records), encoding="utf-8")
        return len(records)
    monkeypatch.setattr("jobmatch.ingest.download_ads", download)
    return tmp_path / "index", tmp_path / "ads.jsonl", records, encoded, clock


def test_refresh_replaces_changed_and_removed_ads_and_reuses_unchanged_embeddings(public_data):
    index, source, records, encoded, clock = public_data
    first = refresh_jobs(index, source)
    clock.now += timedelta(hours=6)
    records[:] = [raw_ad("1", application_deadline="2026-12-31T23:59:59"),
                  raw_ad("2", text="Develop a new Java application."), raw_ad("4", text="Build ML systems.")]
    encoded.clear()
    second = refresh_jobs(index, source)
    current = Retriever(index)
    assert {ad["ad_id"] for ad in current.ads} == {"1", "2", "4"}
    assert next(ad for ad in current.ads if ad["ad_id"] == "2")["text"] == "Develop a new Java application."
    assert next(ad for ad in current.ads if ad["ad_id"] == "1")["deadline"] == "2026-12-31T23:59:59"
    assert len(encoded) == 2  # A deadline-only change does not need a new embedding.
    assert second["generation"] != first["generation"]
    assert second["source"] == str(source.resolve()) and second["scope"] == "tech"
    assert second["ads_fingerprint"] == ads_fingerprint(load_ads(source))
    assert set(current.collection.get()["ids"]) == {"1:0", "2:0", "4:0"}
    assert (index / first["generation"] / "ads.json").exists()


def test_unchanged_feed_updates_check_time_without_rebuilding(public_data):
    index, source, records, encoded, clock = public_data
    first = refresh_jobs(index, source)
    clock.now += timedelta(hours=6)
    records.reverse()
    records[0]["unused_field"] = "does not affect searchable content"
    encoded.clear()
    with patch("jobmatch.retrieval.build_index", side_effect=AssertionError("Unchanged jobs must not rebuild")):
        second = refresh_jobs(index, source)
    assert second["generation"] == first["generation"] and second["built_at"] == first["built_at"]
    assert second["checked_at"] > first["checked_at"]
    assert not encoded


@pytest.mark.parametrize("old_version", [None, TECH_FILTER_VERSION - 1])
def test_filter_change_refreshes_recent_database_then_reuses_six_hour_cache(public_data, old_version):
    from jobmatch import ingest
    index, source, records, _, _ = public_data
    previous = refresh_jobs(index, source)
    previous["filter_version"] = old_version
    (index / "manifest.json").write_text(json.dumps(previous))
    records.append(raw_ad("thesis", "Master Thesis", "Develop machine learning models."))
    with patch("jobmatch.ingest.download_ads", wraps=ingest.download_ads) as download:
        current = refresh_jobs(index, source)
        assert current["filter_version"] == TECH_FILTER_VERSION
        assert refresh_jobs(index, source) == current
        download.assert_called_once()
    retriever = Retriever(index)
    assert {ad["opportunity_type"] for ad in retriever.ads} == {"job", "thesis"}
    for mode in ["keyword", "semantic", "hybrid"]:
        results = retriever.retrieve("machine learning", filters={"opportunity_type": "thesis"}, mode=mode, min_similarity=0)
        assert results and all(c["metadata"]["ad_id"] == "thesis" for c in results)

@pytest.mark.parametrize("failure", ["network", "build", "invalid_data"])
def test_failed_refresh_preserves_working_dataset_and_index(public_data, monkeypatch, failure):
    index, source, records, _, clock = public_data
    refresh_jobs(index, source)
    clock.now += timedelta(hours=6)
    before_manifest = (index / "manifest.json").read_bytes()
    before_data = source.read_bytes()
    records[:] = [raw_ad("new", text="A new role.")]
    def fail(*args, **kwargs): raise OSError("offline or build failed")
    if failure == "network":
        monkeypatch.setattr("jobmatch.ingest.download_ads", fail)
    elif failure == "build":
        monkeypatch.setattr("jobmatch.retrieval.build_index", fail)
    else:
        monkeypatch.setattr("jobmatch.ingest.download_ads", lambda output, *a, **kw: output.write_text("invalid JSON"))
    with pytest.raises((OSError, ValueError)):
        refresh_jobs(index, source)
    assert (index / "manifest.json").read_bytes() == before_manifest
    assert source.read_bytes() == before_data
    assert {ad["ad_id"] for ad in Retriever(index).ads} == {"1", "2", "3"}
    assert not list(source.parent.glob(".jobmatch-refresh-*"))


@pytest.mark.parametrize("elapsed,fetch", [
    (timedelta(0), False),
    (timedelta(hours=6) - timedelta(microseconds=1), False),
    (timedelta(hours=6), True),
    (timedelta(hours=7), True),
])
def test_six_hour_window_uses_persisted_success_time(public_data, elapsed, fetch):
    from jobmatch import ingest
    index, source, _, encoded, clock = public_data
    first = refresh_jobs(index, source)
    before = (index / "manifest.json").read_bytes()
    clock.now += elapsed
    encoded.clear()
    with patch("jobmatch.ingest.download_ads", wraps=ingest.download_ads) as download:
        second = refresh_jobs(index, source)
    assert download.call_count == int(fetch)
    assert second["generation"] == first["generation"] and not encoded
    if fetch:
        assert second["checked_at"] > first["checked_at"]
    else:
        assert second == first and (index / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("timestamp", [None, "invalid", "2026-10-19T12:00:00", "2026-10-20T12:00:00+00:00"])
def test_missing_invalid_or_future_timestamps_cannot_suppress_an_update(public_data, timestamp):
    from jobmatch import ingest
    index, source, _, _, clock = public_data
    first = refresh_jobs(index, source)
    first["checked_at"] = timestamp
    (index / "manifest.json").write_text(json.dumps(first))
    with patch("jobmatch.ingest.download_ads", wraps=ingest.download_ads) as download:
        current = refresh_jobs(index, source)
    download.assert_called_once()
    assert datetime.fromisoformat(current["checked_at"]) == clock.now


def test_update_age_handles_timezones_and_the_daylight_saving_change(public_data):
    index, source, _, _, clock = public_data
    manifest = refresh_jobs(index, source)
    # 02:30 summer time is 00:30 UTC, before Stockholm's clock change.
    manifest["checked_at"] = "2026-10-25T02:30:00+02:00"
    (index / "manifest.json").write_text(json.dumps(manifest))
    # 07:29:59 winter time is 06:29:59 UTC: just under six elapsed hours.
    clock.now = datetime.fromisoformat("2026-10-25T07:29:59+01:00")
    with patch("jobmatch.ingest.download_ads", side_effect=AssertionError("Cache is still fresh")):
        assert refresh_jobs(index, source)["checked_at"] == manifest["checked_at"]
    clock.now += timedelta(seconds=1)
    from jobmatch import ingest
    with patch("jobmatch.ingest.download_ads", wraps=ingest.download_ads) as download:
        refresh_jobs(index, source)
    download.assert_called_once()


@pytest.mark.parametrize("missing", ["ads.json", "chunks.json"])
def test_recent_timestamp_does_not_hide_a_missing_index_file(public_data, missing):
    from jobmatch import ingest
    index, source, _, _, _ = public_data
    first = refresh_jobs(index, source)
    (index / first["generation"] / missing).unlink()
    with patch("jobmatch.ingest.download_ads", wraps=ingest.download_ads) as download:
        second = refresh_jobs(index, source)
    download.assert_called_once()
    assert second["generation"] != first["generation"]
    assert len(Retriever(index).ads) == 3
