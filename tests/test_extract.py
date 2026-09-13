import hashlib

import pytest

from rag.extract import extract_pdf


def blocks_of(doc, page_number, kind):
    page = next(p for p in doc.pages if p.number == page_number)
    return [b for b in page.blocks if b.kind == kind]


def test_pages_are_one_based_and_complete(extracted):
    assert [p.number for p in extracted.pages] == [1, 2]
    assert extracted.source_name == "sample.pdf"


def test_document_id_is_content_hash(sample_pdf, extracted):
    assert extracted.document_id == hashlib.sha256(sample_pdf).hexdigest()[:16]


def test_headings_have_clean_titles_and_levels(extracted):
    headings = [(b.text, b.level) for b in blocks_of(extracted, 1, "heading")]
    assert headings == [("Annual Report", 1), ("1. Introduction", 2), ("2. Financials", 2)]
    assert [b.text for b in blocks_of(extracted, 2, "heading")] == ["3. Outlook"]


def test_table_is_markdown(extracted):
    (table,) = blocks_of(extracted, 1, "table")
    assert "|Year|Revenue|Profit|" in table.text
    assert "|2025|120|15|" in table.text


def test_figure_and_caption_detected_with_bbox(extracted):
    (figure,) = blocks_of(extracted, 1, "figure")
    assert len(figure.bbox) == 4 and figure.bbox[2] > figure.bbox[0]
    (caption,) = blocks_of(extracted, 1, "caption")
    assert caption.text.startswith("Figure 1")


def test_largest_image_area_recorded_per_page(extracted):
    assert extracted.pages[0].largest_image_area > 20_000  # the figure image
    assert extracted.pages[1].largest_image_area == 0


def test_page_text_stays_on_its_page(extracted):
    page1 = " ".join(b.text for b in extracted.pages[0].blocks)
    page2 = " ".join(b.text for b in extracted.pages[1].blocks)
    assert "Outlook sentence" not in page1
    assert "Outlook sentence 2" in page2


@pytest.mark.parametrize("data", [b"", b"not a pdf"])
def test_invalid_pdf_raises_value_error(data):
    with pytest.raises(ValueError, match="Not a readable PDF"):
        extract_pdf(data, "bad.pdf")
