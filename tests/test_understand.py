import json
import re

import pymupdf
import pytest
from qdrant_client import QdrantClient

from rag.chunk import chunk_document
from rag.config import EMBED_DIM, QDRANT_COLLECTION
from rag.embed import EmbeddingCache
from rag.extract import Block, Document, Page, extract_pdf
from rag.ingest import ingest_pdf
from rag.store import iter_payloads
from rag.understand import (
    PagePlan,
    UnderstandingCache,
    merge_page,
    parse_result,
    plan_page,
    understand_document,
)

PROSE = "Employees receive 24 days of paid annual leave per calendar year, with carry-forward rules."
BAD = chr(0xFFFD)  # what PyMuPDF emits for glyphs without a Unicode mapping
BIG = (50, 300, 300, 450)  # 250 x 150 pt
LOGO = (500, 20, 560, 50)  # 60 x 30 pt


def blk(kind, text="", bbox=(50, 50, 400, 70), level=0):
    return Block(kind, text, bbox, level)


class FakeM3:
    """Answers every task found in the prompt, or raises / returns a fixed reply."""

    def __init__(self, reply=None):
        self.reply, self.calls = reply, []

    def __call__(self, messages):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply is not None:
            return self.reply
        tasks = messages[1]["content"][0]["text"]
        result = {
            "page_text": "Transcribed heading\n\nTranscribed paragraph." if "page_text:" in tasks else None,
            "figures": [{"id": int(i), "description": f"Chart {i}: revenue rising"} for i in re.findall(r"figures id (\d+)", tasks)],
            "tables": [{"id": int(i), "markdown": "|Year|Revenue|\n|---|---|\n|2025|120|"} for i in re.findall(r"tables id (\d+)", tasks)],
        }
        return "<think>looking at the page</think>\n```json\n" + json.dumps(result) + "\n```"


# --- Detection ------------------------------------------------------------------


def test_normal_text_page_is_not_sent():
    page = Page(1, [blk("heading", "Leave", level=1), blk("text", PROSE), blk("table", "|a|b|\n|---|---|\n|1|2|"), blk("figure", "", LOGO)])
    assert plan_page(page) == PagePlan()
    assert not plan_page(page).needed


def test_blank_page_is_not_sent():
    assert not plan_page(Page(1, [])).needed
    assert not plan_page(Page(1, [], largest_image_area=1_800)).needed  # small logo only


def test_image_only_page_without_blocks_is_scanned():
    page = Page(3, [], largest_image_area=500_990)
    plan = plan_page(page)
    assert plan.transcribe and plan.reasons == ("scanned",)
    merged = merge_page(page, plan, {"page_text": "Invoice 42\n\nTotal due: 900"})
    assert merged.number == 3 and merged.largest_image_area == 500_990
    assert [(b.kind, b.text) for b in merged.blocks] == [("figure", "Invoice 42"), ("figure", "Total due: 900")]


def test_large_figure_is_described_but_logo_is_not():
    plan = plan_page(Page(1, [blk("text", PROSE), blk("figure", "", LOGO), blk("figure", "", BIG)]))
    assert plan.figures == (2,) and not plan.transcribe and plan.reasons == ("figure",)


def test_scanned_page_is_transcribed_not_described():
    plan = plan_page(Page(1, [blk("figure", "", (0, 0, 612, 792))]))
    assert plan.transcribe and plan.figures == () and plan.reasons == ("scanned",)


def test_short_caption_with_chart_is_described_not_transcribed():
    plan = plan_page(Page(1, [blk("heading", "Growth", level=2), blk("figure", "", BIG), blk("caption", "Figure 1")]))
    assert not plan.transcribe and plan.figures == (1,) and plan.reasons == ("figure",)


def image_only_pdf() -> bytes:
    source = pymupdf.open()
    text_page = source.new_page()
    for i in range(12):
        text_page.insert_text((72, 90 + i * 22), f"Scanned invoice line {i}: amount due 900 rupees.", fontsize=13)
    scan = pymupdf.open()
    page = scan.new_page(width=text_page.rect.width, height=text_page.rect.height)
    page.insert_image(page.rect, pixmap=text_page.get_pixmap(dpi=100))  # page is now just an image
    return scan.tobytes()


def test_real_image_only_page_is_detected_as_scanned():
    doc = extract_pdf(image_only_pdf(), "scan.pdf")
    assert plan_page(doc.pages[0]).reasons == ("scanned",)


def test_image_only_pdf_page_becomes_figure_chunks_via_mocked_m3():
    data = image_only_pdf()
    doc = extract_pdf(data, "scan.pdf")
    m3 = FakeM3('{"page_text": "ACICs in India\\n\\nTotal ACICs: 17\\n\\n| State | ACICs |\\n|---|---|\\n| Tamil Nadu | 3 |"}')
    enriched, report = understand_document(doc, data, m3)

    assert report.understood == [1] and len(m3.calls) == 1
    assert m3.calls[0][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    chunks = chunk_document(enriched)
    assert [(c.content_type, c.text) for c in chunks] == [
        ("figure", "ACICs in India"),
        ("figure", "Total ACICs: 17"),
        ("table", "| State | ACICs |\n|---|---|\n| Tamil Nadu | 3 |"),
    ]
    assert all((c.document_id, c.source_name, c.page_number) == (doc.document_id, "scan.pdf", 1) for c in chunks)


def test_ordinary_text_pages_never_call_m3():
    pdf = pymupdf.open()
    for n in range(3):
        page = pdf.new_page()
        for i in range(10):
            page.insert_text((72, 80 + i * 18), f"Page {n} line {i}: employees receive paid leave and allowances.", fontsize=11)
    data = pdf.tobytes()
    doc = extract_pdf(data, "text.pdf")
    m3 = FakeM3(RuntimeError("M3 must not be called for text pages"))
    enriched, report = understand_document(doc, data, m3)
    assert m3.calls == [] and report.understood == [] and report.failed == []
    assert enriched is doc


def test_garbled_text_is_transcribed_and_figures_still_described():
    garbled = f"क{BAD}मचार{BAD}या{BAD} को {BAD}ावसाय{BAD}क या{BAD}ा " * 3
    plan = plan_page(Page(1, [blk("text", garbled), blk("figure", "", BIG)]))
    assert plan.transcribe and plan.figures == (1,) and plan.reasons == ("garbled", "figure")


def test_sparse_table_is_re_extracted_dense_table_is_not():
    sparse = "|Year|Revenue|Profit|\n|---|---|---|\n|2024| | |\n| | |15|"
    dense = "|Year|Revenue|\n|---|---|\n|2024|100|"
    plan = plan_page(Page(1, [blk("text", PROSE), blk("table", dense), blk("table", sparse)]))
    assert plan.tables == (2,) and plan.reasons == ("table",)


def test_real_garbled_devanagari_extraction_is_detected():
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_htmlbox(pymupdf.Rect(72, 72, 520, 400), "<h2>यात्रा भत्ता नीति</h2><p>कर्मचारियों को व्यावसायिक यात्रा के लिए प्रतिदिन 3000 रुपये का भत्ता मिलता है।</p>")
    doc = extract_pdf(pdf.tobytes(), "hi.pdf")
    assert "garbled" in plan_page(doc.pages[0]).reasons


def test_sample_pdf_selects_only_the_figure_page(extracted):
    plans = [plan_page(p) for p in extracted.pages]
    assert [p.needed for p in plans] == [True, False]
    figure_index = next(i for i, b in enumerate(extracted.pages[0].blocks) if b.kind == "figure")
    assert plans[0].figures == (figure_index,) and not plans[0].tables and not plans[0].transcribe


# --- Merge --------------------------------------------------------------------


def test_figure_description_merged_and_other_blocks_untouched():
    page = Page(5, [blk("heading", "Growth", level=2), blk("figure", "axis label", BIG), blk("caption", "Figure 1: Revenue")])
    plan = plan_page(page)
    merged = merge_page(page, plan, {"figures": [{"id": 1, "description": "Line chart rising from 100 to 120."}]})
    assert merged.number == 5
    assert merged.blocks[1] == Block("figure", "Line chart rising from 100 to 120.\naxis label", BIG)
    assert merged.blocks[0] is page.blocks[0] and merged.blocks[2] is page.blocks[2]
    assert page.blocks[1].text == "axis label"  # original not mutated


def test_transcription_replaces_garbled_text_keeps_described_figure():
    page = Page(4, [blk("heading", f"या{BAD}ा", level=2), blk("text", f"क{BAD}मचार{BAD} " * 10), blk("figure", "", BIG), blk("figure", "", LOGO)])
    plan = plan_page(page)
    result = {
        "page_text": "यात्रा भत्ता नीति\n\nकर्मचारियों को 3000 रुपये मिलते हैं।\n| शहर | भत्ता |\n|---|---|\n| दिल्ली | 3000 |",
        "figures": [{"id": 2, "description": "रेलगाड़ी की तस्वीर"}],
    }
    merged = merge_page(page, plan, result)
    assert merged.number == 4
    assert [(b.kind, b.text) for b in merged.blocks] == [
        ("text", "यात्रा भत्ता नीति"),
        ("text", "कर्मचारियों को 3000 रुपये मिलते हैं।"),
        ("table", "| शहर | भत्ता |\n|---|---|\n| दिल्ली | 3000 |"),
        ("figure", "रेलगाड़ी की तस्वीर"),
    ]


def test_scanned_page_content_becomes_figure_blocks_replacing_the_scan_image():
    page = Page(2, [blk("figure", "", (0, 0, 612, 792))])
    merged = merge_page(page, plan_page(page), {"page_text": "Invoice 42\n\nTotal due: 900\n\n| Item | Rs |\n|---|---|\n| Pump | 900 |"})
    assert [(b.kind, b.text) for b in merged.blocks] == [
        ("figure", "Invoice 42"),
        ("figure", "Total due: 900"),
        ("table", "| Item | Rs |\n|---|---|\n| Pump | 900 |"),
    ]


@pytest.mark.parametrize(
    "caption",
    [
        "Glimpses from workshop on Strengthening PACS held at Pune",
        "Hon’ble Prime Minister with CMs/LGs of States/UTs at the 10th Governing Council Meeting",
        "Dr. Hanif Qureshi, Additional Secretary, addressed a media briefing on the PLI Auto Scheme.",
        "राज्यों में आयोजित संपूर्णता अभियान सम्मान समारोह की झलकियाँ",
    ],
)
def test_event_photo_pages_are_not_sent_to_m3(caption):
    page = Page(1, [blk("text", PROSE), blk("figure", "", BIG), blk("caption", caption)])
    assert not plan_page(page).needed


def test_chart_or_map_with_informational_caption_is_sent():
    page = Page(1, [blk("text", PROSE), blk("figure", "", BIG), blk("caption", "ACICs in India: state-wise distribution")])
    assert plan_page(page).figures == (1,)


def test_scanned_page_keeps_document_page_and_section_metadata():
    doc = Document("doc9", "annual.pdf", [Page(1, [blk("heading", "Annexure-I", level=1), blk("text", PROSE)]), Page(2, [], largest_image_area=500_990)])
    plan = plan_page(doc.pages[1])
    merged = merge_page(doc.pages[1], plan, {"page_text": "Organogram of the Ministry\n\nShri A, Joint Secretary"})
    chunks = chunk_document(Document(doc.document_id, doc.source_name, [doc.pages[0], merged]))
    figure_chunks = [c for c in chunks if c.content_type == "figure"]
    assert [c.text for c in figure_chunks] == ["Organogram of the Ministry", "Shri A, Joint Secretary"]
    for c in figure_chunks:
        assert (c.document_id, c.source_name, c.page_number, c.section_path) == ("doc9", "annual.pdf", 2, ("Annexure-I",))


def test_sparse_table_replaced_keeping_position():
    table = blk("table", "|Year|Revenue|\n|---|---|\n| | |", (40, 200, 500, 300))
    page = Page(3, [blk("text", PROSE), table])
    merged = merge_page(page, plan_page(page), {"tables": [{"id": 1, "markdown": "|Year|Revenue|\n|---|---|\n|2025|120|"}]})
    assert merged.blocks[1] == Block("table", "|Year|Revenue|\n|---|---|\n|2025|120|", (40, 200, 500, 300))


def test_parse_result_tolerates_think_and_code_fences():
    assert parse_result('<think>{"not": "this"}</think>\n```json\n{"page_text": null, "figures": []}\n```') == {
        "page_text": None,
        "figures": [],
    }


# --- Orchestration: selection, fallback, cache, metadata ---------------------------


def test_only_selected_pages_sent_and_merged_with_metadata_intact(extracted, sample_pdf):
    m3 = FakeM3()
    doc, report = understand_document(extracted, sample_pdf, m3)

    assert len(m3.calls) == 1 and report.understood == [1] and report.failed == []
    text_part, image_part = m3.calls[0][1]["content"]
    assert text_part["text"].startswith("Page 1. Tasks:") and "figures id" in text_part["text"]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    assert doc.document_id == extracted.document_id and doc.source_name == extracted.source_name
    assert [p.number for p in doc.pages] == [1, 2]
    assert doc.pages[1] is extracted.pages[1]  # untouched page is the same object
    assert not any("revenue rising" in b.text for b in extracted.pages[0].blocks)  # input not mutated

    figure_chunk = next(c for c in chunk_document(doc) if c.content_type == "figure")
    assert figure_chunk.page_number == 1 and figure_chunk.section == "2. Financials"
    assert figure_chunk.text.startswith("Figure 1: Revenue growth chart") and "revenue rising" in figure_chunk.text


@pytest.mark.parametrize(
    "reply",
    [
        RuntimeError("MiniMax completion failed: HTTP 503"),
        "I cannot help with that.",
        "[]",
        '{"figures": [{"id": 999, "description": "wrong block"}]}',
        '{"figures": "not a list", "tables": null}',
    ],
    ids=["api-error", "no-json", "json-not-object", "unknown-id", "wrong-types"],
)
def test_failures_fall_back_to_fast_extraction(extracted, sample_pdf, reply, caplog):
    doc, report = understand_document(extracted, sample_pdf, FakeM3(reply))
    assert report.understood == [] and report.failed == [1]
    assert doc.pages == extracted.pages
    assert "keeping fast extraction" in caplog.text


def test_successful_results_are_cached_failures_are_not(extracted, sample_pdf, tmp_path):
    cache = UnderstandingCache(tmp_path / "u.sqlite")
    understand_document(extracted, sample_pdf, FakeM3(RuntimeError("down")), cache)

    first = FakeM3()
    doc1, _ = understand_document(extracted, sample_pdf, first, cache)
    assert len(first.calls) == 1  # the failure was not cached

    second = FakeM3(RuntimeError("must not be called"))
    doc2, report = understand_document(extracted, sample_pdf, second, cache)
    assert second.calls == [] and report.understood == [1] and doc2 == doc1
    cache.close()


def test_document_without_candidate_pages_makes_no_calls():
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), PROSE)
    data = pdf.tobytes()
    doc = extract_pdf(data, "plain.pdf")
    m3 = FakeM3()
    result, report = understand_document(doc, data, m3)
    assert result is doc and report.understood == [] and report.failed == []
    assert m3.calls == []


# --- Ingestion integration ----------------------------------------------------------


def fake_embed(texts):
    return [[1.0, float(len(t))] + [0.5] * (EMBED_DIM - 2) for t in texts]


@pytest.fixture
def qdrant():
    client = QdrantClient(":memory:")
    yield client
    client.close()


def test_ingest_stores_m3_figure_description_with_citation_metadata(qdrant, sample_pdf, tmp_path):
    result = ingest_pdf(
        sample_pdf, "sample.pdf", client=qdrant, embed_batch=fake_embed, cache=EmbeddingCache(tmp_path / "e.sqlite"),
        complete=FakeM3(), understanding_cache=UnderstandingCache(tmp_path / "u.sqlite"),
    )
    assert result.understood_pages == [1] and result.failed_pages == []
    figure = next(p for p in iter_payloads(qdrant) if p["content_type"] == "figure")
    assert "revenue rising" in figure["text"]
    assert figure["page_number"] == 1 and figure["section_path"][-1] == "2. Financials"
    assert figure["document_id"] == result.document_id and figure["source_name"] == "sample.pdf"


def test_ingest_survives_m3_outage_with_fast_extraction(qdrant, sample_pdf, tmp_path):
    fast = ingest_pdf(sample_pdf, "sample.pdf", client=QdrantClient(":memory:"), embed_batch=fake_embed, cache=EmbeddingCache(tmp_path / "a.sqlite"))
    degraded = ingest_pdf(
        sample_pdf, "sample.pdf", client=qdrant, embed_batch=fake_embed, cache=EmbeddingCache(tmp_path / "b.sqlite"),
        complete=FakeM3(RuntimeError("gateway down")),
    )
    assert degraded.failed_pages == [1] and degraded.understood_pages == []
    assert degraded.chunks == fast.chunks == qdrant.count(QDRANT_COLLECTION).count
