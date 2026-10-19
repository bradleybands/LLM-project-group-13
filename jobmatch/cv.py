"""Local PDF text extraction with per-page Tesseract OCR fallback."""
import io
import os
from pathlib import Path
import shutil
import subprocess
import threading
from dataclasses import dataclass, field

from jobmatch.config import ROOT

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PAGES = 12
MAX_CV_CHARS = 16000
_PDFIUM_LOCK = threading.RLock()


class CVExtractionError(ValueError):
    """A safe, user-facing extraction error without document contents."""


@dataclass
class CVExtraction:
    text: str
    page_count: int
    ocr_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def ocr_environment() -> dict:
    env = os.environ.copy()
    local = ROOT / ".cache" / "tessdata"
    data_dir = os.getenv("OCR_TESSDATA") or (str(local) if (local / "eng.traineddata").exists() else None)
    if data_dir:
        env["TESSDATA_PREFIX"] = data_dir
    return env


def tesseract_path() -> str | None:
    configured = os.getenv("TESSERACT_PATH")
    if configured:
        return configured if Path(configured).is_file() else None
    return shutil.which("tesseract") or next((str(p) for p in [
        Path("/opt/homebrew/bin/tesseract"), Path("/usr/local/bin/tesseract")
    ] if p.is_file()), None)


def ocr_languages(binary: str) -> tuple[str, list[str]]:
    try:
        result = subprocess.run([binary, "--list-langs"], capture_output=True, timeout=10, env=ocr_environment())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CVExtractionError("Tesseract could not start. Check your local OCR installation.") from exc
    if result.returncode:
        raise CVExtractionError("Tesseract language data could not be loaded.")
    available = set(result.stdout.decode("utf-8", errors="replace").splitlines()[1:])
    requested = (os.getenv("OCR_LANGUAGE") or "eng+swe").split("+")
    selected = [lang for lang in requested if lang in available]
    if not selected and "eng" in available:
        selected = ["eng"]
    if not selected:
        raise CVExtractionError("Install a Tesseract OCR language pack, such as English (eng).")
    missing = [lang for lang in requested if lang not in available]
    warnings = [f"OCR language pack(s) {', '.join(missing)} unavailable; using {'+'.join(selected)}. Review accented words."] if missing else []
    return "+".join(selected), warnings


def ocr_image(png: bytes, binary: str, languages: str) -> str:
    try:
        result = subprocess.run([binary, "stdin", "stdout", "-l", languages, "--psm", "3"],
                                input=png, capture_output=True, timeout=60, env=ocr_environment())
    except subprocess.TimeoutExpired as exc:
        raise CVExtractionError("OCR took too long on a page. Try a clearer or smaller scan.") from exc
    except OSError as exc:
        raise CVExtractionError("The local OCR engine could not start.") from exc
    if result.returncode:
        raise CVExtractionError("OCR could not read a page. Try a clearer scan or paste the CV text.")
    return result.stdout.decode("utf-8", errors="replace").strip()


def render_page(pdf_bytes: bytes, page_number: int) -> bytes:
    import pypdfium2 as pdfium
    with _PDFIUM_LOCK:
        document = pdfium.PdfDocument(pdf_bytes)
        try:
            page = document[page_number]
            try:
                width, height = page.get_size()
                if width <= 0 or height <= 0:
                    raise CVExtractionError("The PDF has an invalid page size.")
                scale = min(3.0, (12_000_000 / (width * height)) ** 0.5)
                bitmap = page.render(scale=scale)
                try:
                    image = bitmap.to_pil()
                    output = io.BytesIO()
                    image.save(output, format="PNG")
                    return output.getvalue()
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            document.close()


def extract_cv_pdf(pdf_bytes: bytes, force_ocr: bool = False) -> CVExtraction:
    from pypdf import PdfReader
    if not pdf_bytes or b"%PDF-" not in pdf_bytes[:1024]:
        raise CVExtractionError("Please upload a valid PDF file.")
    if len(pdf_bytes) > MAX_FILE_BYTES:
        raise CVExtractionError("The PDF is too large. Upload a file smaller than 10 MB.")
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if reader.is_encrypted and not reader.decrypt(""):
            raise CVExtractionError("This PDF is password-protected. Upload an unlocked copy.")
        page_count = len(reader.pages)
    except CVExtractionError:
        raise
    except Exception as exc:
        raise CVExtractionError("This PDF could not be opened. Export a fresh PDF and try again.") from exc
    if not 1 <= page_count <= MAX_PAGES:
        raise CVExtractionError("Upload a CV with 1–12 pages.")
    pages, ocr_pages, warnings = [], [], []
    binary, languages = None, None
    for index, page in enumerate(reader.pages):
        try:
            native = (page.extract_text() or "").strip()
        except Exception:
            native = ""
        needs_ocr = force_ocr or sum(c.isalnum() for c in native) < 80
        text = native
        if needs_ocr:
            try:
                if binary is None:
                    binary = tesseract_path()
                    if not binary:
                        raise CVExtractionError("Scanned PDFs need Tesseract. Install it with 'brew install tesseract', or paste your CV text.")
                    languages, language_warnings = ocr_languages(binary)
                    warnings.extend(language_warnings)
                png = render_page(pdf_bytes, index)
                recognized = ocr_image(png, binary, languages)
                if recognized:
                    text = recognized
                    ocr_pages.append(index + 1)
                else:
                    warnings.append(f"Page {index + 1}: OCR found no readable text. Check this page manually.")
            except CVExtractionError as exc:
                if not native:
                    raise
                warnings.append(f"Page {index + 1}: {exc} Kept the existing PDF text.")
            except Exception as exc:
                raise CVExtractionError("A PDF page could not be rendered for OCR. Try exporting the PDF again.") from exc
        pages.append(text)
    text = "\n\n".join(p for p in pages if p.strip()).strip()
    if not text:
        raise CVExtractionError("No readable CV text was found. Try a clearer scan or paste the text.")
    if len(text) > MAX_CV_CHARS:
        text = text[:MAX_CV_CHARS]
        warnings.append("The extracted CV exceeded 16,000 characters and was shortened. Review the text before matching.")
    return CVExtraction(text, page_count, ocr_pages, list(dict.fromkeys(warnings)))
