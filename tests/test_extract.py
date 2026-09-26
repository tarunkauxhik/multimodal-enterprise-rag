import hashlib

import pytest

from rag.extract import extract_pdf, is_legacy_hindi_font, legacy_font_share, normalize_font_name


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


# --- Legacy Hindi fonts ---------------------------------------------------------


def test_font_names_are_normalised_without_subset_prefix():
    assert normalize_font_name("ABCDEF+BHARTIYA-HINDI_081") == "bhartiyahindi081"
    assert normalize_font_name("Arjun-Bold") == "arjunbold"
    assert normalize_font_name("Kruti Dev 010") == "krutidev010"
    assert normalize_font_name("AbCdEf+Arjun") == "abcdefarjun"  # only a real 6-capital subset tag is stripped


@pytest.mark.parametrize(
    "name",
    # the MHI report's fonts, plus other common Indian government legacy families
    ["Arjun", "Arjun-Bold", "Arjun-Italic", "Arjun-BoldItalic", "BHARTIYA-HINDI_081", "QWERTY+Arjun",
     "Kruti Dev 010", "DevLys 010", "Walkman-Chanakya905", "Chanakya", "Shivaji01", "AkrutiDevPriya"],
)
def test_legacy_hindi_fonts_are_recognised(name):
    assert is_legacy_hindi_font(name)


@pytest.mark.parametrize(
    "name",
    # English fonts from the evaluation reports, real Unicode Devanagari fonts, and names that merely
    # contain a legacy family: matching is anchored to the start of the family name.
    ["NewsGothicBT-Roman", "TimesNewRomanPSMT", "Cambria-Bold", "Helvetica", "Mukta-Regular", "Mangal",
     "NirmalaUI", "SuperArjun", "ABCDEF+TimesArjunSans", "Kruti"],
)
def test_other_fonts_are_not_legacy(name):
    assert not is_legacy_hindi_font(name)


def test_legacy_font_share_counts_latin_letters_per_font():
    spans = [("Arjun", "Hkkjh m|ksx"), ("NewsGothicBT-Roman", "Ministry of")]
    assert legacy_font_share(spans) == pytest.approx(9 / 19)  # "|" and spaces are not letters


def test_legacy_font_share_ignores_digits_so_number_tables_stay_legacy():
    # an MHI-style Hindi table: legacy-font words, numbers set in an English font
    spans = [("ABCDEF+Arjun", "ch,pbZ,y lhlhvkbZ"), ("NewsGothicBT-Roman", "686.00 220.33 725.00 2569.00")]
    assert legacy_font_share(spans) == 1.0


def test_legacy_font_share_of_a_page_without_latin_letters_is_zero():
    assert legacy_font_share([]) == 0.0
    assert legacy_font_share([("Mukta-Regular", "भारी उद्योग मंत्रालय 2025")]) == 0.0


def test_generated_english_pdf_has_no_legacy_font_share(extracted):
    assert [p.legacy_font_share for p in extracted.pages] == [0.0, 0.0]
