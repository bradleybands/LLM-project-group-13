"""Display original job-ad structure separately from flattened search text."""
import html
import json
import re
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from jobmatch.data import clean_text


# Some publishers export every line as an ordinary paragraph, including their
# section labels. Recognize labels, rather than treating every short sentence
# or skill name as a heading. The original spelling and language stay intact.
_SECTION_LABELS = frozenset(label.casefold() for label in """
Om rollen
Din roll
Rollen
Om tjänsten
Om jobbet
Om uppdraget
Om uppdragen
Arbetsuppgifter
Dina arbetsuppgifter
Huvudsakliga arbetsuppgifter
Exempel på arbetsuppgifter
Ansvarsområden
Dina ansvarsområden
Kvalifikationer
Dina kvalifikationer
Krav
Kravprofil
Obligatoriska krav
Kompetenskrav
Din profil
Din bakgrund
Bakgrund
Erfarenhet
Utbildning
Meriterande
Det är meriterande om du har
Personliga egenskaper
Vem är du
Vem söker vi
Om dig
Vi söker dig som
Vi söker dig som har
Vi erbjuder
Vi erbjuder dig
Vad vi erbjuder
Du erbjuds
Förmåner
Trygghet & förmåner
Flexibilitet & variation
Om oss
Om företaget
Om arbetsgivaren
Om verksamheten
Ansökan
Din ansökan
Så ansöker du
Välkommen med din ansökan
Varför söka
Kontakt
Kontaktperson
Övrigt
Övrig information
Praktisk information
Anställningsvillkor
About the role
The role
Your role
About the position
About the job
Responsibilities
Your responsibilities
Key responsibilities
What you will do
What you'll do
Requirements
Qualifications
Your qualifications
Required qualifications
Preferred qualifications
Skills and experience
Experience
Education
Your profile
Your background
Who you are
What we are looking for
What we're looking for
What you bring
About you
We offer
What we offer
Benefits
About us
About the company
About the position and the company
You are offered
Examples of work tasks
Additional information
Application
How to apply
Why join us
Contact
""".strip().splitlines())


def _section_headings(body: str) -> str:
    """Give explicit headings and recognizable plain section labels one style."""
    def paragraph(match):
        tag, content = match.groups()
        # Do not reinterpret paragraphs containing lists, tables or other blocks.
        if re.search(r"</?(?:p|div|h[1-6]|ul|ol|li|table|pre|blockquote)\b", content):
            return match.group(0)
        lines = re.split(r"<br\s*/?>", content)
        parts, pending = [], []

        def flush():
            if pending:
                parts.append(f"<{tag}>" + "<br>".join(pending) + f"</{tag}>")
                pending.clear()

        for line in lines:
            label = clean_text(line).rstrip(":? ").casefold()
            bold = re.fullmatch(r"\s*<(strong|b)>\s*((?:(?!</?(?:strong|b)\b).)+?)\s*</\1>\s*", line, re.DOTALL)
            if label in _SECTION_LABELS or bold:
                flush()
                parts.append("<h3>" + (bold.group(2) if bold else line.strip()) + "</h3>")
            else:
                pending.append(line)
        flush()
        return "".join(parts)

    return re.sub(r"<(p|div)>((?:(?!</?(?:p|div)\b).)*?)</\1>", paragraph, body, flags=re.DOTALL)


class _AdHTML(HTMLParser):
    """Keep document structure, without embedded code, styling or images."""

    allowed = {"p", "br", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6",
               "ul", "ol", "li", "strong", "b", "em", "i", "u", "blockquote", "pre", "code",
               "a", "hr", "table", "thead", "tbody", "tr", "th", "td", "caption"}
    hidden = {"script", "style", "iframe", "object", "svg", "math", "template"}
    void = {"br", "hr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.hidden:
            self.hidden_depth += 1
        if self.hidden_depth or tag not in self.allowed:
            return
        attributes = ""
        if tag == "a":
            href = dict(attrs).get("href") or ""
            try:
                scheme = urlparse(href.strip()).scheme.lower()
            except ValueError:
                scheme = ""
            if scheme in {"http", "https", "mailto"}:
                attributes = f' href="{html.escape(href.strip(), quote=True)}" target="_blank" rel="noopener noreferrer"'
        if tag in {"ol", "li"}:
            name = "start" if tag == "ol" else "value"
            number = dict(attrs).get(name) or ""
            if re.fullmatch(r"-?\d+", number):
                attributes = f' {name}="{number}"'
            if tag == "ol" and "reversed" in dict(attrs):
                attributes += " reversed"
        self.parts.append(f"<{tag}{attributes}>")

    def handle_endtag(self, tag):
        if self.hidden_depth:
            if tag in self.hidden:
                self.hidden_depth -= 1
            return
        if tag in self.allowed and tag not in self.void:
            self.parts.append(f"</{tag}>")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.void:
            self.handle_endtag(tag)

    def handle_data(self, data):
        if not self.hidden_depth:
            self.parts.append(html.escape(data, quote=False))


def description_html(description: dict | str) -> str:
    """Render the publisher's formatting and wording without rewriting it."""
    if isinstance(description, dict):
        formatted = description.get("text_formatted") or ""
        plain = description.get("text") or ""
    else:
        formatted, plain = "", str(description or "")
    if formatted.strip():
        parser = _AdHTML()
        parser.feed(formatted)
        parser.close()
        body = "".join(parser.parts)
        if clean_text(body):
            return _section_headings(body)
    # A plain-text-only ad still keeps its original paragraphs and line breaks.
    plain = plain.replace("\r\n", "\n").replace("\r", "\n")
    return _section_headings("".join("<p>" + html.escape(paragraph).replace("\n", "<br>") + "</p>"
                                    for paragraph in re.split(r"\n\s*\n", plain) if paragraph.strip()))


@lru_cache(maxsize=64)
def _saved_description(source: str, modified_ns: int, size: int, ad_id: str) -> str:
    # File revision arguments invalidate the cache after an atomic feed update.
    with Path(source).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if isinstance(raw, dict) and str(raw.get("id")) == ad_id:
                return description_html(raw.get("description") or "")
    return ""


def original_ad_html(ad: dict, source: str | Path | None = None) -> str:
    """Read original formatting from the saved feed, keeping search text separate."""
    if source:
        path = Path(source)
        try:
            revision = path.stat()
            body = _saved_description(str(path.resolve()), revision.st_mtime_ns, revision.st_size, ad["ad_id"])
            if body:
                return body
        except (OSError, ValueError, TypeError):
            # Older/moved indexes can still display their stored plain text.
            pass
    return description_html(ad.get("text", ""))
