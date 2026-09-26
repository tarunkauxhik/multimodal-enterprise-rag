"""Fast PDF extraction with PyMuPDF4LLM (layout mode).

Produces typed blocks in reading order, grouped by page. No network calls:
OCR is disabled here; scanned pages and figures are left for the selective
MiniMax understanding step, which fills in figure block text before chunking.
"""

import hashlib
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import pymupdf
import pymupdf4llm

HEADING_CLASSES = {"title", "section-header"}
FIGURE_CLASSES = {"picture", "formula"}
SKIP_CLASSES = {"page-header", "page-footer"}

# Legacy (pre-Unicode) Hindi fonts draw Devanagari glyphs from Latin code points, so their text
# extracts as Latin gibberish ("Hkkjh m|ksx" for भारी उद्योग) with no U+FFFD. Matched against the
# start of the normalised family name, never as a substring.
# ponytail: closed list of common Indian government fonts; unknown ones fall back to understand's text fingerprint
LEGACY_HINDI_FONTS = (
    "krutidev",
    "devlys",
    "chanakya",
    "walkmanchanakya",
    "shivaji",
    "arjun",
    "bhartiyahindi",
    "akruti",
    "apsdv",
    "shusha",
)

_HEADING_MD = re.compile(r"^(#+)\s*(.*)$", re.S)
_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")  # embedded-subset tag, e.g. "ABCDEF+Arjun"


@dataclass
class Block:
    kind: str  # "heading" | "text" | "table" | "figure" | "caption"
    text: str  # markdown; may be empty for figures without extractable text
    bbox: tuple[int, int, int, int]  # page coordinates, for cropping/highlighting
    level: int = 0  # heading depth (1 = top); 0 for non-headings


@dataclass
class Page:
    number: int  # 1-based
    blocks: list[Block]
    # Largest embedded image on the page (pt², clipped to the page). Layout mode returns no
    # blocks at all for image-only (scanned) pages, so this is how they are recognised.
    largest_image_area: int = 0
    # Share of the page's Latin letters drawn in a legacy Hindi font, read from the PDF's own
    # text spans (PyMuPDF4LLM's blocks carry no font names).
    legacy_font_share: float = 0.0


@dataclass
class Document:
    document_id: str  # content hash: re-uploading the same PDF gives the same id
    source_name: str
    pages: list[Page]


def document_id_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def extract_pdf(data: bytes, source_name: str) -> Document:
    """Extract a PDF (raw bytes, e.g. a Streamlit upload) into pages of blocks."""
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except pymupdf.FileDataError as exc:
        raise ValueError(f"Not a readable PDF: {source_name}") from exc

    with doc:
        if doc.needs_pass:
            raise ValueError(f"Encrypted PDF not supported: {source_name}")
        page_chunks = pymupdf4llm.to_markdown(
            doc,
            page_chunks=True,
            use_ocr=False,
            header=False,
            footer=False,
            show_progress=False,
        )
        image_areas = [_largest_image_area(page) for page in doc]
        legacy_shares = [legacy_font_share(_text_spans(page)) for page in doc]

    pages = []
    for c in page_chunks:
        number = c["metadata"]["page_number"]
        pages.append(Page(number, _blocks(c), image_areas[number - 1], legacy_shares[number - 1]))
    return Document(document_id_for(data), source_name, pages)


def _largest_image_area(page: pymupdf.Page) -> int:
    areas = [(pymupdf.Rect(info["bbox"]) & page.rect).get_area() for info in page.get_image_info()]
    return round(max(areas, default=0))


def normalize_font_name(name: str) -> str:
    """'ABCDEF+BHARTIYA-HINDI_081' -> 'bhartiyahindi081': no subset tag, lowercase, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", _SUBSET_PREFIX.sub("", name).lower())


def is_legacy_hindi_font(name: str) -> bool:
    return normalize_font_name(name).startswith(LEGACY_HINDI_FONTS)


def legacy_font_share(spans: Iterable[tuple[str, str]]) -> float:
    """Share of Latin letters drawn in a legacy Hindi font, over (font name, text) spans.

    Letters only: digits and punctuation are font-neutral, so a legacy-font table full of
    numbers set in an English font still reads as the legacy page it is.
    """
    total = legacy = 0
    for font, text in spans:
        letters = sum(ch.isascii() and ch.isalpha() for ch in text)
        total += letters
        if letters and is_legacy_hindi_font(font):
            legacy += letters
    return legacy / total if total else 0.0


def _text_spans(page: pymupdf.Page) -> Iterator[tuple[str, str]]:
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", ()):  # image blocks have no lines
            for span in line["spans"]:
                yield span["font"], span["text"]


def _blocks(page_chunk: dict) -> list[Block]:
    text = page_chunk["text"]
    blocks = []
    for box in page_chunk["page_boxes"]:
        cls = box["class"]
        if cls in SKIP_CLASSES:
            continue
        start, stop = box["pos"]
        raw = text[start:stop].strip()
        bbox = tuple(box["bbox"])

        if cls in FIGURE_CLASSES:
            blocks.append(Block("figure", raw, bbox))  # kept even if empty: caption/MiniMax may fill it
        elif not raw:
            continue
        elif cls in HEADING_CLASSES:
            m = _HEADING_MD.match(raw)
            level = len(m.group(1)) if m else 1
            title = " ".join((m.group(2) if m else raw).replace("**", "").split())
            blocks.append(Block("heading", title, bbox, level))
        elif cls == "table":
            blocks.append(Block("table", raw, bbox))
        elif cls == "caption":
            blocks.append(Block("caption", raw, bbox))
        else:  # text, list-item, footnote, ...
            blocks.append(Block("text", raw, bbox))
    return blocks
