import { render, screen, within } from "@testing-library/react"
import { describe as group, expect, it } from "vitest"

import { Passage } from "@/components/sources/passage"
import { FIGURE_CAPTION, parsePassage } from "@/lib/passage"

group("source text: meaningful text stays, artifacts go, HTML stays inert", () => {
  it.each([
    ["2*3", "2*3"],
    ["C++", "C++"],
    ["A/B", "A/B"],
    ["______", "______"],
    ["x^2 + y_1 = z", "x^2 + y_1 = z"],
    ["See https://example.com/a_b_c now", "See https://example.com/a_b_c now"],
    ["<sup>literal text</sup>", "<sup>literal text</sup>"], // document content, not formatting
    ["<script>alert(1)</script>", "<script>alert(1)</script>"],
    ["**B.TECH. VI**<sup>**th**</sup> **SEMESTER**", "B.TECH. VIth SEMESTER"],
  ])("%s", (input, visible) => {
    const { container } = render(<Passage text={input} />)
    expect(container).toHaveTextContent(visible, { normalizeWhitespace: true })
    expect(container.querySelector("script, a, img, iframe, b, strong")).toBeNull()
  })

  it("drops a bare *** rule instead of showing it", () => {
    const { container } = render(<Passage text={"Above\n***\nBelow"} />)
    expect(container.textContent).not.toContain("*")
    expect(container).toHaveTextContent("Above")
    expect(container).toHaveTextContent("Below")
  })

  it("renders <sup> as an element but any other tag as literal text", () => {
    const { container } = render(<Passage text={"x<sup>2</sup> <em>not</em> <div>markup</div>"} />)
    expect(container.querySelector("sup")).toHaveTextContent("2")
    expect(container.querySelector("em")).toBeNull()
    expect(container).toHaveTextContent("<em>not</em> <div>markup</div>")
    expect(container.innerHTML).toContain("&lt;div&gt;markup&lt;/div&gt;") // escaped text, not an element
  })
})

group("malformed tables: valid structure becomes a table, anything else stays readable", () => {
  it("pads missing cells and keeps inconsistent rows aligned", () => {
    const [block] = parsePassage("|a|b|c|\n|---|---|---|\n|1|2|\n|3|4|5|6|")
    expect(block).toEqual({ type: "table", header: ["a", "b", "c", ""], rows: [["1", "2", "", ""], ["3", "4", "5", "6"]] })
  })

  it("keeps empty cells, maths, code-like text and URLs as plain cell text", () => {
    const text = String.raw`|Formula|Code|Link|
|---|---|---|
|x^2 + y_1|fn(a \| b)|https://example.com/x_y|
||empty first||`
    render(<Passage contentType="table" text={text} />)
    const table = screen.getByRole("table")
    expect(within(table).getByText("x^2 + y_1")).toBeInTheDocument()
    expect(within(table).getByText("fn(a | b)")).toBeInTheDocument()
    expect(within(table).getByText("https://example.com/x_y").closest("a")).toBeNull()
    expect(within(table).getAllByRole("row")).toHaveLength(3)
  })

  it("falls back to text when the structure is broken, without losing content", () => {
    const blocks = parsePassage("| only one pipe row |\n| a | b\nplain line")
    expect(blocks).toEqual([{ type: "text", text: "| only one pipe row |\n| a | b\nplain line" }])
    expect(parsePassage("|---|---|\n|---|---|")).toEqual([]) // separators only: nothing to show
  })

  it("renders long cells and very long tables inside a scrolling container", () => {
    const rows = Array.from({ length: 500 }, (_, i) => `|${i + 1}|${"a very long cell ".repeat(20)}|${i * 3}|`).join("\n")
    const { container } = render(<Passage contentType="table" text={`|#|Text|Value|\n|---|---|---|\n${rows}`} />)
    expect(screen.getAllByRole("row")).toHaveLength(501)
    expect(container.querySelector(".data-table")).toHaveClass("overflow-x-auto", "max-w-full")
  })
})

group("figure captions are promoted conservatively", () => {
  it.each(["Figure 1: Revenue growth chart", "Fig. 2.1 – Regional map", "Chart IV. Sales by quarter", "Exhibit 3a: Structure"])("promotes %s", (line) => {
    expect(FIGURE_CAPTION.test(line)).toBe(true)
  })

  it.each(["Image quality matters: poor scans lose detail", "Chart: revenue", "Figures show strong growth in 2025.", "Figure", `Figure 1: ${"x".repeat(300)}`])(
    "leaves %s as text",
    (line) => {
      expect(FIGURE_CAPTION.test(line)).toBe(false)
    },
  )

  it("only titles figure chunks, never text chunks", () => {
    render(<Passage contentType="text" text={"Figure 1: Revenue growth chart\nThe chart shows growth."} />)
    expect(screen.getByText(/Figure 1: Revenue growth chart/)).not.toHaveClass("font-medium")
  })
})

group("superscripts: only known extraction shapes are formatting", () => {
  it("renders VI<sup>th</sup> as VIᵗʰ with a real superscript", () => {
    const { container } = render(<Passage text="B.TECH. VI<sup>th</sup> SEMESTER" />)
    expect(container.querySelectorAll("sup")).toHaveLength(1)
    expect(container.querySelector("sup")).toHaveTextContent("th")
    expect(container).toHaveTextContent("B.TECH. VIth SEMESTER")
    expect(container.textContent).not.toContain("<sup>")
  })

  it.each(["x<sup>2</sup>", "1<sup>st</sup> place", "10<sup>-3</sup> m", "Note<sup>*</sup>", "Brand<sup>®</sup>"])("formats %s", (text) => {
    const { container } = render(<Passage text={text} />)
    expect(container.querySelector("sup")).not.toBeNull()
  })

  it("keeps H<sub>2</sub>O and log<sub>10</sub> as subscripts", () => {
    const { container } = render(<Passage text="H<sub>2</sub>O and log<sub>10</sub>" />)
    expect([...container.querySelectorAll("sub")].map((s) => s.textContent)).toEqual(["2", "10"])
  })

  it.each([
    "<sup>literal text</sup>", // not attached to a word
    "word<sup>literal text</sup>", // attached, but not a typographic mark
    "a <sup>2</sup>", // detached by a space
    "x<sup><b>2</b></sup>", // nested tags
    "H<sub>not a subscript</sub>",
  ])("keeps %s as inert literal text", (text) => {
    const { container } = render(<Passage text={text} />)
    expect(container.querySelector("sup, sub, b")).toBeNull()
    expect(container).toHaveTextContent(text, { normalizeWhitespace: true })
  })
})
