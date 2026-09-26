// Source passages are extracted document text in which tables are Markdown pipe tables
// (PyMuPDF4LLM). They are shown as real tables, but every cell is still plain text: nothing in a
// passage is interpreted as HTML or Markdown formatting.

export type PassageBlock = { type: "text"; text: string } | { type: "table"; header: string[] | null; rows: string[][] }

const TABLE_LINE = /^\s*\|.*\|\s*$/
const SEPARATOR = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/

function cells(line: string): string[] {
  const parts = line.trim().replace(/^\|/, "").replace(/\|$/, "").split(/(?<!\\)\|/)
  return parts.map((cell) => cell.replace(/\\\|/g, "|").replace(/<br\s*\/?>/gi, "\n").trim())
}

export function parsePassage(text: string): PassageBlock[] {
  const blocks: PassageBlock[] = []
  const lines = text.split("\n")
  let prose: string[] = []
  const flushProse = () => {
    const joined = prose.join("\n").trim()
    if (joined) blocks.push({ type: "text", text: joined })
    prose = []
  }
  for (let i = 0; i < lines.length; ) {
    if (!TABLE_LINE.test(lines[i])) {
      prose.push(lines[i++])
      continue
    }
    const start = i
    while (i < lines.length && TABLE_LINE.test(lines[i])) i++
    const rows = lines.slice(start, i)
    if (rows.length < 2) {
      prose.push(...rows) // a single pipe line is just text
      continue
    }
    flushProse()
    const hasHeader = SEPARATOR.test(rows[1])
    const body = rows.filter((row) => !SEPARATOR.test(row)).map(cells)
    const width = Math.max(...body.map((row) => row.length))
    const padded = body.map((row) => [...row, ...Array(width - row.length).fill("")])
    blocks.push(hasHeader ? { type: "table", header: padded[0], rows: padded.slice(1) } : { type: "table", header: null, rows: padded })
  }
  flushProse()
  return blocks
}

/** One line of readable text for a source row: table cells joined, whitespace collapsed. */
export function passagePreview(text: string, max = 160): string {
  const flat = parsePassage(text)
    .map((block) =>
      block.type === "text" ? block.text : [block.header, ...block.rows].filter(Boolean).map((row) => row!.filter(Boolean).join(" · ")).join(" · "),
    )
    .join(" ")
    .replace(/\s+/g, " ")
    .trim()
  return flat.length > max ? `${flat.slice(0, max - 1).trimEnd()}…` : flat
}
