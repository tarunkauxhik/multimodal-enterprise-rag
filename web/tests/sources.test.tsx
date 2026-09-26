import { render, screen, within } from "@testing-library/react"
import { describe as group, expect, it } from "vitest"

import { Passage } from "@/components/sources/passage"
import { cleanText, inline, parsePassage, plain } from "@/lib/passage"

group("source text cleanup", () => {
  it("removes bold markers and keeps super/subscripts as structure", () => {
    const text = cleanText("**B.TECH. VI**<sup>**th**</sup> **SEMESTER**")
    expect(plain(text)).toBe("B.TECH. VIth SEMESTER")
    expect(inline(text)).toEqual([{ text: "B.TECH. VI" }, { text: "th", style: "sup" }, { text: " SEMESTER" }])
    expect(inline(cleanText("H<sub>2</sub>O"))).toEqual([{ text: "H" }, { text: "2", style: "sub" }, { text: "O" }])
  })

  it("turns tables with escaped bold markers into clean cells", () => {
    const [block] = parsePassage(String.raw`| **\*\*S No\*\*\*\*TOPIC** | **PAGE** |
|---|---|
| 1 | SI & CI |
| 2 | Coding-Decoding |`)
    expect(block).toEqual({ type: "table", header: ["S No TOPIC", "PAGE"], rows: [["1", "SI & CI"], ["2", "Coding-Decoding"]] })
  })

  it("recognises a bold first row as a header and drops empty columns", () => {
    expect(parsePassage("|**Year**|**Revenue**||\n|2025|120||")).toEqual([{ type: "table", header: ["Year", "Revenue"], rows: [["2025", "120"]] }])
  })

  it("is conservative: real asterisks, blanks and maths stay", () => {
    expect(cleanText("Name: ______ and 2*3=6, a * b, footnote*")).toBe("Name: ______ and 2*3=6, a * b, footnote*")
  })

  it("keeps link text but never its target, and drops images", () => {
    expect(cleanText("[click here](javascript:alert(1)) ![logo](x.png) done")).toBe("click here done")
  })

  it("finds headings and bullets", () => {
    expect(parsePassage("## Scope\n- first *point*\n* second")).toEqual([
      { type: "heading", text: "Scope" },
      { type: "text", text: "• first point\n• second" },
    ])
  })
})

group("source rendering", () => {
  it("shows B.TECH. VIth SEMESTER with a real superscript and no markup", () => {
    const { container } = render(<Passage text="**B.TECH. VI**<sup>**th**</sup> **SEMESTER**" />)
    expect(container).toHaveTextContent("B.TECH. VIth SEMESTER")
    expect(container.querySelector("sup")).toHaveTextContent("th")
    expect(container.textContent).not.toMatch(/\*|<sup>|<\/sup>/)
  })

  it("renders tables as tables without pipes, backslashes or bold markers", () => {
    const { container } = render(<Passage contentType="table" text={String.raw`| **\*\*S No\*\*** | **TOPIC** | **PAGE** |
|---|---|---|
| 1 | SI & CI | 3–11 |`} />)
    const table = screen.getByRole("table")
    expect(within(table).getAllByRole("columnheader").map((th) => th.textContent)).toEqual(["S No", "TOPIC", "PAGE"])
    expect(container.textContent).not.toMatch(/[|\\*]/)
    expect(table.parentElement).toHaveClass("overflow-x-auto", "max-w-full") // long tables scroll inside, not the page
  })

  it("keeps HTML and script-like document text inert", () => {
    const { container } = render(<Passage text={'<script>alert(1)</script><img src=x onerror="alert(2)"> <b>bold</b> [x](javascript:alert(3))'} />)
    expect(container.querySelector("script, img, b, a, iframe")).toBeNull()
    expect(container).toHaveTextContent('<script>alert(1)</script><img src=x onerror="alert(2)"> <b>bold</b> x')
  })

  it("titles a figure passage with its caption", () => {
    render(<Passage contentType="figure" text={"Figure 1: Revenue growth chart\nA bar chart of revenue by year."} />)
    expect(screen.getByText("Figure 1: Revenue growth chart")).toHaveClass("font-medium")
    expect(screen.getByText("A bar chart of revenue by year.")).toBeInTheDocument()
  })
})
