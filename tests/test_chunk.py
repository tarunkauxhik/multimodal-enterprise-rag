from rag.chunk import chunk_document
from rag.extract import Block, Document, Page

BOX = (0, 0, 10, 10)


def h(text, level):
    return Block("heading", text, BOX, level)


def b(kind, text):
    return Block(kind, text, BOX)


def doc(*pages):
    return Document("doc123", "manual.pdf", [Page(i + 1, blocks) for i, blocks in enumerate(pages)])


def test_section_path_nests_resets_and_carries_across_pages():
    chunks = chunk_document(
        doc(
            [h("Guide", 1), h("Setup", 2), b("text", "Install it."), h("Usage", 2), b("text", "Run it.")],
            [b("text", "Still usage."), h("Appendix", 1), b("text", "Extra.")],
        )
    )
    got = [(c.page_number, c.section_path, c.text) for c in chunks]
    assert got == [
        (1, ("Guide", "Setup"), "Install it."),
        (1, ("Guide", "Usage"), "Run it."),
        (2, ("Guide", "Usage"), "Still usage."),
        (2, ("Appendix",), "Extra."),
    ]
    assert chunks[0].section == "Setup"


def test_chunks_carry_document_metadata_and_unique_ids():
    chunks = chunk_document(doc([b("text", "a")], [b("text", "b")]))
    assert all(c.document_id == "doc123" and c.source_name == "manual.pdf" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_table_and_figure_are_standalone_with_captions():
    chunks = chunk_document(
        doc(
            [
                b("text", "Intro."),
                b("caption", "Table 2: Costs"),
                b("table", "|a|b|\n|---|---|\n|1|2|"),
                b("figure", ""),
                b("caption", "Figure 3: Trend"),
                b("figure", ""),  # no text, no caption -> nothing to index yet
                b("text", "Outro."),
            ]
        )
    )
    got = [(c.content_type, c.text) for c in chunks]
    assert got == [
        ("text", "Intro."),
        ("table", "Table 2: Costs\n|a|b|\n|---|---|\n|1|2|"),
        ("figure", "Figure 3: Trend"),
        ("text", "Outro."),
    ]


def test_long_prose_splits_under_limit_without_losing_words():
    words = [f"word{i}" for i in range(400)]
    chunks = chunk_document(doc([b("text", " ".join(words))]), max_chars=300)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == words


def test_small_prose_blocks_pack_together():
    chunks = chunk_document(doc([b("text", "One."), b("text", "Two.")]))
    assert [c.text for c in chunks] == ["One.\n\nTwo."]


def test_large_table_splits_by_rows_repeating_header():
    header = "|id|name|\n|---|---|"
    rows = [f"|{i}|name-{i}|" for i in range(100)]
    chunks = chunk_document(doc([b("table", "\n".join([header, *rows]))]), max_chars=300)
    assert len(chunks) > 1
    assert all(c.text.startswith(header) and len(c.text) <= 300 for c in chunks)
    body = [line for c in chunks for line in c.text.splitlines()[2:]]
    assert body == rows


def test_real_pdf_end_to_end(extracted):
    chunks = chunk_document(extracted)
    by_type = {c.content_type: c for c in chunks}
    assert by_type["table"].page_number == 1
    assert by_type["table"].section == "2. Financials"
    assert by_type["figure"].text.startswith("Figure 1")
    outlook = [c for c in chunks if c.page_number == 2]
    assert outlook and all(c.section == "3. Outlook" for c in outlook)
    assert all(c.document_id == extracted.document_id for c in chunks)
