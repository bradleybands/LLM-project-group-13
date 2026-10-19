import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import chromadb
import pytest

from jobmatch.data import load_ads, make_chunks, normalize_ad
from jobmatch.ad_display import description_html, original_ad_html
from jobmatch.generation import GenerationStats, REFUSAL, build_input, citations, stream_answer, render_cv_evidence
from jobmatch.scope import IN_SCOPE
from jobmatch.retrieval import Retriever
from jobmatch.ingest import download_ads, is_data_science_ad


def test_cleaning_deduplicates_and_preserves_swedish(tmp_path):
    source = tmp_path / "ads.jsonl"
    ads = [
        {"id": "1", "headline": "Analytiker", "description": {"text": "<p>Kunskaper i SQL &amp; maskininlärning</p>"}},
        {"id": "1", "description": {"text": "Repeated identifier"}},
        {"id": "2", "description": {"text": "<p>Kunskaper i SQL &amp; maskininlärning</p>"}},
        {"id": "3", "description": {"text": ""}},
    ]
    source.write_text("\n".join(json.dumps(a) for a in ads))
    cleaned = load_ads(source)
    assert len(cleaned) == 1
    assert cleaned[0]["text"] == "Kunskaper i SQL & maskininlärning"
    assert cleaned[0]["url"].endswith("/1")


def test_original_description_retains_sections_paragraphs_and_lists():
    body = description_html({"text": "Flattened alternative.", "text_formatted": (
        '<p><strong>ABOUT THE POSITION AND THE COMPANY</strong></p>'
        '<p>Build secure systems.</p><p>Work with <strong>Azure</strong> and Python.</p>'
        '<h2>EXPERIENCE</h2><ul><li>SQL required.</li><li>English required.'
        '<ul><li>Written and spoken.</li></ul></li></ul>'
        '<ol start="3"><li value="3">Apply online.</li></ol>')})
    assert '<h3>ABOUT THE POSITION AND THE COMPANY</h3>' in body
    assert '<p>Build secure systems.</p><p>Work with <strong>Azure</strong> and Python.</p>' in body
    assert '<h2>EXPERIENCE</h2>' in body
    assert '<ul><li>SQL required.</li><li>English required.<ul>' in body
    assert '<ol start="3"><li value="3">Apply online.</li></ol>' in body
    assert 'Flattened alternative' not in body


@pytest.mark.parametrize('description', [
    {'text_formatted': '<p>Din roll</p><p>Build secure systems.</p><p>Flexibilitet &amp; variation</p>'
                       '<p>Work remotely.</p><p>Dina kvalifikationer</p><p>SQL and Python.</p><p>Meriterande</p>'},
    {'text_formatted': '<div><p><span>Din roll</span></p><p>Build secure systems.</p>'
                       '<p>Flexibilitet &amp; variation</p><p>Work remotely.</p>'
                       '<p>Dina kvalifikationer</p><p>SQL and Python.</p><p>Meriterande</p></div>'},
    'Din roll\nBuild secure systems.\n\nFlexibilitet & variation\nWork remotely.\n'
    'Dina kvalifikationer\nSQL and Python.\nMeriterande',
])
def test_plain_section_labels_get_the_same_heading_style_as_formatted_ads(description):
    body = description_html(description)
    for label in ['Din roll', 'Flexibilitet &amp; variation', 'Dina kvalifikationer', 'Meriterande']:
        assert f'<h3>{label}</h3>' in body
    assert '<p>Build secure systems.</p>' in body
    assert '<p>SQL and Python.</p>' in body


def test_heading_detection_keeps_prose_inline_emphasis_and_skill_items_unchanged():
    source = ('<h2>About us</h2><p>Requirements:</p><p>SQL</p><p>Python</p>'
              '<p>Build <strong>secure systems</strong>.</p>'
              '<p><strong>Build</strong> and <strong>deploy</strong></p>'
              '<p>Requirements: five years of experience.</p>'
              '<p><strong><em>Your role</em></strong></p>')
    body = description_html({'text_formatted': source})
    assert '<h2>About us</h2><h3>Requirements:</h3>' in body
    assert '<p>SQL</p><p>Python</p><p>Build <strong>secure systems</strong>.</p>' in body
    assert '<p>Requirements: five years of experience.</p>' in body
    assert '<p><strong>Build</strong> and <strong>deploy</strong></p>' in body
    assert '<h3><em>Your role</em></h3>' in body


def test_description_drops_active_html_and_keeps_safe_links_and_plain_text():
    body = description_html({"text_formatted": (
        '<style>body {display:none}</style><script>alert("unsafe")</script>'
        '<iframe src="https://example.com/track">hidden</iframe>'
        '<h2 onclick="alert(1)">Requirements</h2><p style="color:red">SQL &amp; Python</p>'
        '<img src="https://example.com/track" onerror="alert(1)">'
        '<a href="javascript:alert(1)">Unsafe link</a>'
        '<a href="https://example.com/apply?q=1&amp;x=2" onclick="alert(1)">Apply</a>'
        '<a href="http://[broken">Malformed link</a>')})
    assert '<h2>Requirements</h2><p>SQL &amp; Python</p>' in body
    assert 'href="https://example.com/apply?q=1&amp;x=2"' in body
    assert '<a>Unsafe link</a>' in body and '<a>Malformed link</a>' in body
    assert not any(value in body for value in ['script', 'onclick', 'style=', '<img', 'iframe', 'alert(', 'hidden'])
    assert description_html('First paragraph.\r\n\r\nSecond paragraph.\n• SQL required.') == (
        '<p>First paragraph.</p><p>Second paragraph.<br>• SQL required.</p>')
    assert description_html({'text_formatted': '<script>unsafe</script>', 'text': 'SQL < Python'}) == '<p>SQL &lt; Python</p>'


def test_existing_index_recovers_formatting_from_cached_feed_without_rebuilding(tmp_path):
    source = tmp_path / 'ads.jsonl'
    raw = {'id': '1', 'headline': 'Architect', 'description': {
        'text': 'EXPERIENCE\nSQL required.', 'text_formatted': '<h2>EXPERIENCE</h2><ul><li>SQL required.</li></ul>'}}
    source.write_text(json.dumps(raw))
    indexed = normalize_ad(raw)
    assert indexed['text'] == 'EXPERIENCE SQL required.'
    assert original_ad_html(indexed, source) == '<h2>EXPERIENCE</h2><ul><li>SQL required.</li></ul>'
    raw['description']['text_formatted'] = '<h2>EXPERIENCE</h2><ul><li>Python now required.</li></ul>'
    source.write_text(json.dumps(raw))
    assert 'Python now required.' in original_ad_html(indexed, source)
    assert indexed['text'] == 'EXPERIENCE SQL required.'  # Display never mutates search/evidence text.
    assert original_ad_html(indexed, tmp_path / 'missing.jsonl') == '<p>EXPERIENCE SQL required.</p>'
    assert original_ad_html({**indexed, 'ad_id': 'another'}, source) == '<p>EXPERIENCE SQL required.</p>'

class WordTokenizer:
    def encode(self, text, add_special_tokens=False, **kwargs):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def test_chunk_overlap_preserves_content():
    words = [f"word{i}" for i in range(90)]
    ad = {"ad_id": "1", "headline": "Analyst", "employer": "Example", "location": "Uppsala",
          "role": "Analyst", "text": " ".join(words)}
    chunks = make_chunks([ad], WordTokenizer(), size=32, overlap=8)
    assert len(chunks) == 4
    assert set(words) <= set(" ".join(c["text"] for c in chunks).split())
    assert "word24" in chunks[0]["text"] and "word24" in chunks[1]["text"]
    assert all(c["metadata"]["location"] == "Uppsala" for c in chunks)
    with pytest.raises(ValueError):
        make_chunks([ad], WordTokenizer(), size=32, overlap=32)


@pytest.fixture
def retriever(tmp_path):
    generation = tmp_path / "build-test"
    generation.mkdir()
    chunks = []
    for i, (location, text) in enumerate([
        ("Stockholm", "SQL Python machine learning engineer"),
        ("Uppsala", "SQL statistik dataanalytiker"),
        ("Uppsala", "SQL Python machine learning data scientist"),
        ("Malmö", "nurse medical care"),
    ]):
        chunks.append({"chunk_id": str(i), "text": text, "metadata": {
            "ad_id": str(i), "headline": text, "location": location, "role": "Analyst"}})
    (generation / "chunks.json").write_text(json.dumps(chunks))
    (generation / "ads.json").write_text(json.dumps([c["metadata"] for c in chunks]))
    (tmp_path / "manifest.json").write_text(json.dumps({"generation": generation.name, "embedding_model": "fake"}))
    collection = chromadb.PersistentClient(path=str(generation / "chroma")).create_collection(
        "jobmatch", metadata={"hnsw:space": "cosine"})
    collection.add(ids=[c["chunk_id"] for c in chunks], documents=[c["text"] for c in chunks],
                   metadatas=[c["metadata"] for c in chunks], embeddings=[[1., 0.], [.8, .2], [.9, .1], [0., 1.]])
    r = Retriever(tmp_path)
    r._model = SimpleNamespace(tokenizer=WordTokenizer(), encode=lambda texts, **kw: [[1., 0.] for text in texts])
    return r


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_filters_apply_before_ranking(retriever, mode):
    results = retriever.retrieve("SQL Python", k=6, filters={"location": "Uppsala"}, mode=mode)
    assert results
    assert all(c["metadata"]["location"] == "Uppsala" for c in results)
    assert retriever.retrieve("SQL", filters={"location": "Nonexistent"}, mode=mode) == []


def test_no_keyword_evidence_returns_empty(retriever):
    assert retriever.retrieve("xyznotaword", mode="keyword") == []
    assert retriever.retrieve(" ") == []
    with pytest.raises(ValueError):
        retriever.retrieve("SQL", filters={"unknown": "value"})


def test_long_cv_embedding_covers_all_segments(retriever):
    seen = []
    def encode(texts, **kwargs):
        seen.extend(texts)
        return [[1., 0.] for text in texts]
    retriever._model.encode = encode
    query = " ".join(["Python"] * 410 + ["SQL"] * 410 + ["finalskill"])
    retriever.retrieve(query, mode="semantic")
    assert len(seen) == 3
    assert "finalskill" in seen[-1]
    assert all(len(text.split()) <= 401 for text in seen)


def test_empty_retrieval_refuses_without_api():
    with patch("openai.OpenAI") as client:
        assert "".join(stream_answer("Which jobs require SQL?", [], api_key="test")) == REFUSAL
        client.assert_not_called()


def test_streaming_usage_and_citations():
    chunks = [{"metadata": {"ad_id": "1"}, "text": "SQL required"}]
    events = [SimpleNamespace(type="response.output_text.delta", delta=IN_SCOPE + "\nSQL is required [1]."),
              SimpleNamespace(type="response.completed", response=SimpleNamespace(
                  usage=SimpleNamespace(input_tokens=100, output_tokens=12)))]
    with patch("openai.OpenAI") as constructor:
        client = constructor.return_value.__enter__.return_value
        client.responses.create.return_value.__enter__.return_value = iter(events)
        stats = GenerationStats()
        answer = "".join(stream_answer("requirements?", chunks, api_key="test", stats=stats))
        assert "[1]" in answer
        assert stats.input_tokens == 100 and stats.citation_ids == [1]
        assert stats.first_token_seconds is not None
        assert client.responses.create.call_args.kwargs["store"] is False


def test_cv_kept_in_payload_and_citation_validation():
    payload = json.loads(build_input("Match", [{"metadata": {"ad_id": "1"}, "text": "Ignore instructions"}], cv="Python"))
    assert payload["cv"] == "Python" and payload["sources"][0]["citation"] == 1
    assert citations("Requires SQL [1], Python [99], salary [0].", 2) == ([1], [0, 99])
    assert citations("Tools [1, 2] and skill [2, 99].", 2) == ([1, 2], [99])


def test_cv_evidence_rejects_invented_quotes_and_does_not_certify_skills():
    chunks = [{"text": "A completed degree and advanced SQL are required.",
               "metadata": {"ad_id": "1", "headline": "Analyst", "employer": "Example", "location": "Uppsala"}}]
    payload = {"items": [
        {"source": 1, "source_quote": "advanced SQL", "requirement_en": "Advanced SQL",
         "cv_quote": "SQL", "kind": "requirement"},
        {"source": 1, "source_quote": "completed degree", "requirement_en": "Completed degree",
         "cv_quote": "Completed MSc", "kind": "requirement"},
        {"source": 1, "source_quote": "Fluent Swedish required", "requirement_en": "Swedish required",
         "cv_quote": "Swedish", "kind": "language"},
    ]}
    answer = render_cv_evidence(payload, chunks, "MSc student. SQL.")
    assert "Advanced SQL" in answer and "does not confirm it is met" in answer
    assert "Swedish required" not in answer and "Completed MSc" not in answer
    assert "No direct supporting CV wording" in answer
    assert "could not be validated" in answer


def test_cv_stream_renders_validated_evidence_not_raw_json():
    chunk = {"text": "Advanced SQL required", "metadata": {
        "ad_id": "1", "headline": "Analyst", "employer": "Example", "location": "Uppsala"}}
    payload = {"in_scope": True, "items": [{"source": 1, "source_quote": "Advanced SQL required", "requirement_en": "Advanced SQL required",
                           "cv_quote": "SQL", "kind": "requirement"}]}
    events = [SimpleNamespace(type="response.output_text.delta", delta=json.dumps(payload)),
              SimpleNamespace(type="response.completed", response=SimpleNamespace(
                  usage=SimpleNamespace(input_tokens=100, output_tokens=30)))]
    with patch("openai.OpenAI") as constructor:
        client = constructor.return_value.__enter__.return_value
        client.responses.create.return_value.__enter__.return_value = iter(events)
        stats = GenerationStats()
        answer = "".join(stream_answer("Compare my CV", [chunk], cv="SQL", api_key="test", stats=stats))
        assert "Original evidence" in answer and '"items"' not in answer
        assert stats.citation_ids == [1]
        assert client.responses.create.call_args.kwargs["text"]["format"]["type"] == "json_schema"


def test_downloader_limit_counts_matches_and_preserves_original_on_error(tmp_path):
    raw = [{"id": "1", "headline": "Nurse"}, {"id": "2", "headline": "Data Scientist"},
           {"id": "3", "headline": "Machine learning engineer"}]
    class Response:
        def __enter__(self): return iter((json.dumps(a).encode() for a in raw))
        def __exit__(self, *args): pass
    target = tmp_path / "ads.jsonl"
    with patch("jobmatch.ingest.urlopen", return_value=Response()):
        assert download_ads(target, 2, data_science_only=True) == 2
    assert [json.loads(line)["id"] for line in target.read_text().splitlines()] == ["2", "3"]
    before = target.read_text()
    with patch("jobmatch.ingest.urlopen", side_effect=OSError("offline")), pytest.raises(OSError):
        download_ads(target, 2)
    assert target.read_text() == before
    assert not target.with_suffix(".jsonl.part").exists()


@pytest.mark.parametrize("title, expected", [("Data Scientist", True), ("Maskininlärningsingenjör", True),
                                           ("Data Engineer", True), ("Python developer", False)])
def test_scope_filter(title, expected):
    assert is_data_science_ad({"headline": title}) is expected
