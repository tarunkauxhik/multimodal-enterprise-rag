import pymupdf
import pytest


def build_sample_pdf() -> bytes:
    """Two-page PDF with headings, prose, a ruled table, an image and a caption.

    Generated in code so no binary fixture is committed.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    page.insert_text((72, y), "Annual Report", fontsize=24, fontname="hebo")
    y += 40
    page.insert_text((72, y), "1. Introduction", fontsize=16, fontname="hebo")
    y += 24
    for i in range(4):
        page.insert_text((72, y), f"This is paragraph line {i} describing the company results in detail.", fontsize=11)
        y += 16
    y += 20
    page.insert_text((72, y), "2. Financials", fontsize=16, fontname="hebo")
    y += 24
    rows = [["Year", "Revenue", "Profit"], ["2024", "100", "10"], ["2025", "120", "15"]]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(72 + c * 120, y + r * 20, 72 + (c + 1) * 120, y + (r + 1) * 20)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 4, rect.y1 - 6), cell, fontsize=10)
    y += 90
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 80), 0)
    pix.set_rect(pix.irect, (200, 30, 30))
    page.insert_image(pymupdf.Rect(72, y, 272, y + 130), pixmap=pix)
    page.insert_text((72, y + 140), "Figure 1: Revenue growth chart", fontsize=9)

    page2 = doc.new_page()
    page2.insert_text((72, 72), "3. Outlook", fontsize=16, fontname="hebo")
    for i in range(3):
        page2.insert_text((72, 100 + i * 16), f"Outlook sentence {i} about next year plans.", fontsize=11)
    return doc.tobytes()


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    return build_sample_pdf()


def build_text_pdf(lines: list[str]) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(lines):
        page.insert_text((72, 72 + i * 18), line, fontsize=11)
    return doc.tobytes()


def hash_embed(texts):
    """Offline embedder: bag-of-tokens hashed into EMBED_DIM buckets (similar wording -> similar vectors)."""
    import zlib

    from rag.bm25 import tokenize
    from rag.config import EMBED_DIM

    vectors = []
    for text in texts:
        v = [0.0] * EMBED_DIM
        for token in tokenize(text):
            v[zlib.crc32(token.encode()) % EMBED_DIM] += 1.0
        v[0] += 1e-3  # never a zero vector
        vectors.append(v)
    return vectors


class RecordingModel:
    """Fake MiniMax client: returns `reply` (or raises it) and records every call."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, messages):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture
def fake_services(tmp_path):
    from qdrant_client import QdrantClient

    from rag.session import Services

    client = QdrantClient(":memory:")
    yield Services(
        client=client,
        embed_document=hash_embed,
        embed_query=hash_embed,
        rerank=lambda query, docs, top_n: [(i, 1.0 - i / 100) for i in range(len(docs))][:top_n],
        answer_model=RecordingModel("Revenue was 120 in 2025 [sample.pdf, Page 1]."),
        understand_model=RecordingModel(RuntimeError("M3 offline in tests")),
        embed_cache_path=tmp_path / "embeddings.sqlite",
        understanding_cache_path=tmp_path / "understanding.sqlite",
        secrets=("sk-test-secret",),
    )
    client.close()


@pytest.fixture(scope="session")
def extracted(sample_pdf):
    from rag.extract import extract_pdf

    return extract_pdf(sample_pdf, "sample.pdf")
