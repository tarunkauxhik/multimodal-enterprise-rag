"""Structure-aware chunking.

Rules:
- A chunk never spans pages, so every chunk cites exactly one page.
- Headings maintain a section path that carries across pages; a heading
  closes the current prose chunk.
- Tables and figures become their own chunks, with adjacent captions attached.
- Prose is packed up to max_chars; oversized blocks split at whitespace, and
  oversized tables split by rows with the header repeated.
"""

from dataclasses import dataclass

from rag.extract import Block, Document

# ponytail: not benchmarked; tune once retrieval evals exist
DEFAULT_MAX_CHARS = 1500


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    source_name: str
    page_number: int
    section_path: tuple[str, ...]  # e.g. ("Annual Report", "2. Financials")
    content_type: str  # "text" | "table" | "figure"
    text: str

    @property
    def section(self) -> str:
        return self.section_path[-1] if self.section_path else ""


def chunk_document(doc: Document, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    headings: list[tuple[int, str]] = []  # (level, title) stack

    def emit(page_number: int, content_type: str, text: str) -> None:
        text = text.strip()
        if text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.document_id}-p{page_number}-{len(chunks)}",
                    document_id=doc.document_id,
                    source_name=doc.source_name,
                    page_number=page_number,
                    section_path=tuple(title for _, title in headings),
                    content_type=content_type,
                    text=text,
                )
            )

    for page in doc.pages:
        blocks = page.blocks
        owners = _caption_owners(blocks)
        prose: list[str] = []

        def flush() -> None:
            for piece in _pack(prose, max_chars):
                emit(page.number, "text", piece)
            prose.clear()

        for i, block in enumerate(blocks):
            if i in owners:
                continue  # emitted with its table/figure
            if block.kind == "heading":
                flush()
                headings[:] = [h for h in headings if h[0] < block.level] + [(block.level, block.text)]
            elif block.kind in ("table", "figure"):
                flush()
                caption = "\n".join(blocks[c].text for c, owner in owners.items() if owner == i)
                prefix = f"{caption}\n" if caption else ""
                if block.kind == "table":
                    for piece in _split_table(block.text, max(max_chars - len(prefix), 200)):
                        emit(page.number, "table", prefix + piece)
                else:
                    emit(page.number, "figure", prefix + block.text)
            else:  # text, or a caption with no table/figure next to it
                prose.append(block.text)
        flush()

    return chunks


def _caption_owners(blocks: list[Block]) -> dict[int, int]:
    """Map caption index -> index of the adjacent table/figure it describes."""
    owners = {}
    for i, block in enumerate(blocks):
        if block.kind != "caption":
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(blocks) and blocks[j].kind in ("table", "figure"):
                owners[i] = j
                break
    return owners


def _pack(parts: list[str], max_chars: int) -> list[str]:
    out, current = [], ""
    for part in parts:
        for piece in _split_text(part, max_chars):
            if current and len(current) + 2 + len(piece) > max_chars:
                out.append(current)
                current = piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
    if current:
        out.append(current)
    return out


def _split_text(text: str, max_chars: int) -> list[str]:
    pieces = []
    while len(text) > max_chars:
        cut = max(text.rfind(" ", 0, max_chars + 1), text.rfind("\n", 0, max_chars + 1))
        if cut <= 0:
            cut = max_chars
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces


def _split_table(table: str, max_chars: int) -> list[str]:
    if len(table) <= max_chars:
        return [table]
    lines = table.splitlines()
    has_separator = len(lines) > 1 and set(lines[1].strip()) <= set("|-: ")
    header = "\n".join(lines[: 2 if has_separator else 1])
    pieces, rows = [], []
    size = len(header)
    for row in lines[2 if has_separator else 1 :]:
        if rows and size + len(row) + 1 > max_chars:
            pieces.append("\n".join([header, *rows]))
            rows, size = [], len(header)
        rows.append(row)
        size += len(row) + 1
    if rows:
        pieces.append("\n".join([header, *rows]))
    return pieces
