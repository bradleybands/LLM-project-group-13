import io
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from jobmatch.cv import CVExtractionError, extract_cv_pdf, tesseract_path


@pytest.fixture
def native_pdf():
    writer = PdfWriter()
    page = writer.add_blank_page(595, 842)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({
        NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 14 Tf 50 780 Td (SYNTHETIC CV - TEST DATA) Tj 0 -28 Td "
                    b"(Skills: Python SQL Machine Learning) Tj 0 -28 Td "
                    b"(Education: MSc student in Data Science) Tj 0 -28 Td "
                    b"(Projects: Built a classification model.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def scanned_pdf():
    image = Image.new("RGB", (1654, 2339), "white")
    ImageDraw.Draw(image).multiline_text(
        (140, 160), "SYNTHETIC CV - TEST DATA\n\nSkills: Python SQL Machine Learning\n"
        "Education: MSc student in Data Science\nProjects: Built a classification model.",
        font=ImageFont.load_default(size=40), fill="black", spacing=22)
    buffer = io.BytesIO()
    image.save(buffer, "PDF", resolution=200)
    return buffer.getvalue()


def test_native_pdf_needs_no_ocr(native_pdf):
    with patch("jobmatch.cv.ocr_image") as ocr:
        result = extract_cv_pdf(native_pdf)
        ocr.assert_not_called()
    assert "Python SQL" in result.text and "MSc student" in result.text
    assert result.page_count == 1 and result.ocr_pages == []


def test_real_scanned_pdf_ocr(scanned_pdf):
    if not tesseract_path():
        pytest.skip("Install Tesseract to run the live OCR integration test")
    result = extract_cv_pdf(scanned_pdf)
    assert "Python" in result.text and "SQL" in result.text
    assert "MSc" in result.text and result.ocr_pages == [1]


def test_mixed_pdf_ocr_only_scanned_page(native_pdf, scanned_pdf):
    writer = PdfWriter()
    writer.append(io.BytesIO(native_pdf))
    writer.append(io.BytesIO(scanned_pdf))
    output = io.BytesIO()
    writer.write(output)
    with patch("jobmatch.cv.tesseract_path", return_value="tesseract"), \
         patch("jobmatch.cv.ocr_languages", return_value=("eng", [])), \
         patch("jobmatch.cv.ocr_image", return_value="Scanned page SQL and Python") as ocr:
        result = extract_cv_pdf(output.getvalue())
    assert result.page_count == 2 and result.ocr_pages == [2]
    assert "MSc student" in result.text and "Scanned page SQL" in result.text
    ocr.assert_called_once()


def test_force_ocr_and_missing_engine(native_pdf, scanned_pdf):
    with patch("jobmatch.cv.tesseract_path", return_value=None):
        with pytest.raises(CVExtractionError, match="Tesseract"):
            extract_cv_pdf(scanned_pdf)
    with patch("jobmatch.cv.tesseract_path", return_value="tesseract"), \
         patch("jobmatch.cv.ocr_languages", return_value=("eng", [])), \
         patch("jobmatch.cv.ocr_image", return_value="Forced OCR text"):
        result = extract_cv_pdf(native_pdf, force_ocr=True)
    assert result.text == "Forced OCR text" and result.ocr_pages == [1]


def test_invalid_locked_and_oversized_pdf(native_pdf):
    with pytest.raises(CVExtractionError, match="valid PDF"):
        extract_cv_pdf(b"not a PDF")
    with pytest.raises(CVExtractionError, match="too large"):
        extract_cv_pdf(b"%PDF-1.7\n" + b"0" * (10 * 1024 * 1024))
    writer = PdfWriter()
    writer.append(io.BytesIO(native_pdf))
    writer.encrypt("synthetic-test-password")
    output = io.BytesIO()
    writer.write(output)
    with pytest.raises(CVExtractionError, match="password-protected"):
        extract_cv_pdf(output.getvalue())


def test_too_many_pages_are_rejected():
    writer = PdfWriter()
    for i in range(13):
        writer.add_blank_page(595, 842)
    output = io.BytesIO()
    writer.write(output)
    with pytest.raises(CVExtractionError, match="1–12 pages"):
        extract_cv_pdf(output.getvalue())


def test_uploaded_pdf_review_consent_and_replacement(native_pdf, monkeypatch, tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    from streamlit.testing.v1 import AppTest
    import jobmatch.config
    monkeypatch.setattr(jobmatch.config, "INDEX_PATH", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    (tmp_path / "manifest.json").write_text('{"test":"upload"}')
    ad = {"ad_id": "1", "headline": "Analyst", "text": "SQL required", "location": "Uppsala",
          "role": "Analyst", "employer": "Example", "deadline": "", "published": "",
          "url": "https://example.com/ad"}
    chunk = {"chunk_id": "1:0", "text": "SQL required", "metadata": {k: v for k, v in ad.items() if k != "text"}}
    retriever = SimpleNamespace(ads=[ad], manifest={"built_at": "2026-10-06"},
                                filter_ads=lambda filters: [ad], retrieve=lambda *a, **kw: [chunk])
    uploaded = SimpleNamespace(getvalue=lambda: native_pdf)
    def answer(*args, **kwargs):
        assert "Python SQL" in kwargs["cv"]
        kwargs["stats"].citation_ids = [1]
        yield "Related CV wording: SQL [1]."
    with patch("jobmatch.ingest.refresh_jobs"), \
         patch("jobmatch.retrieval.Retriever", return_value=retriever), \
         patch("streamlit.file_uploader", return_value=uploaded), \
         patch("jobmatch.generation.stream_answer", side_effect=answer) as model:
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
        assert not app.exception
        assert "Python SQL" in app.text_area(key="cv").value
        assert not app.checkbox(key="cv_consent").value
        assert next(b for b in app.button if b.label == "Find matching roles").disabled
        model.assert_not_called()
        app.checkbox(key="cv_consent").check().run()
        next(b for b in app.button if b.label == "Find matching roles").click().run()
        assert not app.exception and len(app.session_state["messages"]) == 2
        assert model.call_count == 1
        # A new/bad upload must discard stale CV text and consent.
        uploaded.getvalue = lambda: b"not a PDF"
        app.run()
        assert not app.exception
        assert app.text_area(key="cv").value == ""
        assert not app.checkbox(key="cv_consent").value
        assert not app.session_state["messages"]
        assert any("valid PDF" in error.value for error in app.error)
