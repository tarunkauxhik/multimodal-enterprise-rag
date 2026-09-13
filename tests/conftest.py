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


@pytest.fixture(scope="session")
def extracted(sample_pdf):
    from rag.extract import extract_pdf

    return extract_pdf(sample_pdf, "sample.pdf")
