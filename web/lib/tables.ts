// Numeric columns (amounts, years, percentages) are right-aligned so figures line up.

const NUMERIC = /^[\s(+\-−–]*[$€£₹¥]?\s*\d[\d.,\s]*%?\)?\s*(?:cr|crore|lakh|mn|m|bn|b|k)?\.?$/i
const BLANK = /^[\s\-–—]*$/ // empty cells and dashes do not decide a column

/** For each column, whether every non-blank cell in the body is a number (and at least one is).
 * The first column of a multi-column table is a row label (a year, a name) and stays left-aligned. */
export function numericColumns(rows: string[][]): boolean[] {
  const width = Math.max(0, ...rows.map((row) => row.length))
  return Array.from({ length: width }, (_, col) => {
    if (col === 0 && width > 1) return false
    const cells = rows.map((row) => row[col] ?? "").filter((cell) => !BLANK.test(cell))
    return cells.length > 0 && cells.every((cell) => NUMERIC.test(cell))
  })
}

type Node = { type: string; tagName?: string; value?: string; properties?: Record<string, unknown>; children?: Node[] }

const text = (node: Node): string => (node.type === "text" ? node.value ?? "" : (node.children ?? []).map(text).join(""))
const cellsOf = (row: Node) => (row.children ?? []).filter((c) => c.tagName === "td" || c.tagName === "th")
const rowsIn = (section: Node | undefined) => (section?.children ?? []).filter((c) => c.tagName === "tr")

/** rehype plugin: marks numeric columns of Markdown tables with data-numeric, for right alignment. */
export function rehypeNumericColumns() {
  const walk = (node: Node) => {
    if (node.tagName === "table") {
      const sections = node.children ?? []
      const head = rowsIn(sections.find((s) => s.tagName === "thead"))
      const body = rowsIn(sections.find((s) => s.tagName === "tbody"))
      const numeric = numericColumns(body.map((row) => cellsOf(row).map(text)))
      for (const row of [...head, ...body]) {
        cellsOf(row).forEach((cell, i) => {
          if (numeric[i]) cell.properties = { ...cell.properties, dataNumeric: "" }
        })
      }
      return
    }
    node.children?.forEach(walk)
  }
  return (tree: Node) => walk(tree)
}
