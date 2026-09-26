// Source passages are extracted document text (PyMuPDF4LLM Markdown, or MiniMax-M3 transcriptions)
// shown to people. They are untrusted data: nothing here produces HTML. This module only removes
// extraction artifacts (bold markers, escapes, link syntax) and finds structure (headings, pipe
// tables, super/subscripts); components/sources/passage.tsx renders the result as React text.
//
// Cleanup is deliberately conservative, because real document text can contain asterisks, maths,
// underscores and HTML-looking strings:
// - runs of 2+ asterisks are bold markers (never meaningful in extracted text): removed, or a space
//   when they sit between two words ("S No****TOPIC" -> "S No TOPIC");
// - single *x* / _x_ are unwrapped only as a pair at word boundaries; "______" blanks stay;
// - Markdown escapes (\* \| \_ …) are unescaped; links keep their text, images are dropped;
// - <sup>/<sub> become real super/subscripts only in the extraction shape (VI<sup>th</sup>, H<sub>2</sub>O,
//   see inline()); "<sup>literal text</sup>" and any other tag-like text stay literal text.

export type PassageBlock =
  | { type: "text"; text: string }
  | { type: "heading"; text: string }
  | { type: "table"; header: string[] | null; rows: string[][] }

export type Inline = { text: string; style?: "sup" | "sub" }

const WORD = /[\p{L}\p{N}]/u

/** Removes extraction artifacts from one piece of text. <sup>/<sub> tags are kept for inline(). */
export function cleanText(input: string): string {
  let s = input.replace(/<br\s*\/?>/gi, "\n")
  // A link or image target, allowing one level of nested parentheses: (javascript:alert(1)).
  const target = String.raw`\((?:[^()\n]|\([^()\n]*\))*\)`
  s = s.replace(new RegExp(String.raw`!\[[^\]\n]*\]` + target, "g"), "") // images: nothing to show
  s = s.replace(new RegExp(String.raw`\[([^\]\n]+)\]` + target, "g"), "$1") // links: the text, never the target
  s = s.replace(/\\([\\`*_{}[\]()#+\-.!|~<>])/g, "$1") // Markdown escapes
  s = s.replace(/\*{2,}/g, (run: string, offset: number, all: string) =>
    WORD.test(all[offset - 1] ?? "") && WORD.test(all[offset + run.length] ?? "") ? " " : "",
  )
  s = s.replace(/(^|[\s(["'])([*_])(?=\S)([^*_\n]*?\S)\2(?=$|[\s).,;:!?\]"'])/gm, "$1$3") // *italic* / _italic_
  s = s.replace(/~~(?=\S)([^~\n]*?\S)~~/g, "$1")
  s = s.replace(/`([^`\n]+)`/g, "$1")
  return s
    .split("\n")
    .map((line) => line.replace(/[ \t]{2,}/g, " ").trim())
    .join("\n")
}

// The shapes PyMuPDF4LLM produces for real super/subscripts: a short mark attached directly to the
// preceding word or number (VI<sup>th</sup>, x<sup>2</sup>, H<sub>2</sub>O, log<sub>10</sub>).
const SUP_MARK = /^([+\-−]?\d{1,3}|st|nd|rd|th|[a-z]|\*{1,3}|[†‡§¶]|tm|®)$/i
const SUB_MARK = /^([a-z\d]{1,3})$/i

/** Splits cleaned text into plain runs and super/subscript runs. A <sup>/<sub> tag counts as
 * formatting only in the extraction shape above (attached to a word, short typographic mark);
 * anything else, such as "<sup>literal text</sup>", is document content and stays literal text. */
export function inline(text: string): Inline[] {
  const parts: Inline[] = []
  let last = 0
  for (const match of text.matchAll(/<(sup|sub)>([^<>\n]*)<\/\1>/gi)) {
    const style = match[1].toLowerCase() as "sup" | "sub"
    const attached = match.index > 0 && /[\p{L}\p{N})\]]/u.test(text[match.index - 1])
    if (!attached || !(style === "sup" ? SUP_MARK : SUB_MARK).test(match[2])) continue
    if (match.index > last) parts.push({ text: text.slice(last, match.index) })
    parts.push({ text: match[2], style })
    last = match.index + match[0].length
  }
  if (last < text.length) parts.push({ text: text.slice(last) })
  return parts
}

/** Cleaned text as a plain string (super/subscripts inline), for previews and labels. */
export function plain(text: string): string {
  return inline(text).map((part) => part.text).join("")
}

const TABLE_LINE = /^\s*\|.*\|\s*$/
const SEPARATOR = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/
const HEADING = /^#{1,6}\s+(.+)$/
const BULLET = /^[-*•]\s+/
const BOLD_CELL = /^\s*(\\?\*){2,}.*(\\?\*){2,}\s*$|^\s*$/

function splitCells(line: string): string[] {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split(/(?<!\\)\|/)
}

function table(rows: string[]): PassageBlock {
  const raw = rows.filter((row) => !SEPARATOR.test(row)).map(splitCells)
  // A header is marked by a separator row, or by a first row whose cells are all bold.
  const hasHeader = (rows.length > 1 && SEPARATOR.test(rows[1])) || (raw.length > 1 && raw[0].every((cell) => BOLD_CELL.test(cell)))
  let body = raw.map((row) => row.map(cleanText))
  const width = Math.max(...body.map((row) => row.length))
  body = body.map((row) => [...row, ...Array(width - row.length).fill("")])
  const keep = Array.from({ length: width }, (_, col) => body.some((row) => row[col] !== "")) // drop empty columns
  body = body.map((row) => row.filter((_, col) => keep[col])).filter((row) => row.some((cell) => cell !== ""))
  if (hasHeader && body.length > 0) return { type: "table", header: body[0], rows: body.slice(1) }
  return { type: "table", header: null, rows: body }
}

export function parsePassage(text: string): PassageBlock[] {
  const blocks: PassageBlock[] = []
  const lines = text.replace(/\r\n?/g, "\n").split("\n")
  let prose: string[] = []
  const flushProse = () => {
    const joined = cleanText(prose.join("\n")).trim()
    if (joined) blocks.push({ type: "text", text: joined })
    prose = []
  }
  for (let i = 0; i < lines.length; ) {
    const heading = lines[i].trim().match(HEADING)
    if (heading) {
      flushProse()
      const title = cleanText(heading[1]).trim()
      if (title) blocks.push({ type: "heading", text: title })
      i++
      continue
    }
    if (!TABLE_LINE.test(lines[i])) {
      prose.push(lines[i++].trimStart().replace(BULLET, "• "))
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
    const block = table(rows)
    if (block.type === "table" && (block.rows.length > 0 || block.header)) blocks.push(block)
  }
  flushProse()
  return blocks
}

/** A table described in words ("Table: Year, Revenue · 2 rows"), since flattened cells read as noise. */
function describeTable(block: Extract<PassageBlock, { type: "table" }>): string {
  const rows = `${block.rows.length} ${block.rows.length === 1 ? "row" : "rows"}`
  const columns = (block.header ?? []).map(plain).filter(Boolean).join(", ")
  return columns ? `Table: ${columns} · ${rows}` : `Table · ${rows}`
}

/** One line of readable text for a source row: prose, headings, and tables described in words. */
export function passagePreview(text: string, max = 160): string {
  const flat = parsePassage(text)
    .map((block) => (block.type === "table" ? describeTable(block) : plain(block.text)))
    .join(" ")
    .replace(/\s+/g, " ")
    .trim()
  return flat.length > max ? `${flat.slice(0, max - 1).trimEnd()}…` : flat
}

/** A figure passage's leading caption, shown as its title. Conservative: a caption word followed by a
 * number ("Figure 3:", "Fig. 2.1 –", "Chart IV."), on a short first line of a figure chunk only. */
export const FIGURE_CAPTION = /^(figure|fig\.?|chart|graph|diagram|exhibit|illustration)\s*(\d+(\.\d+)*[a-z]?|[ivxlc]+)\s*[:.–-]\s*\S.{0,150}$/i
