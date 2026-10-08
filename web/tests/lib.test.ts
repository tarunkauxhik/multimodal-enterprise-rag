import { describe as group, expect, it, vi } from "vitest"

import { chatHistory, type Message } from "@/hooks/use-conversations"
import { api, ApiError, errorMessage, HISTORY_ANSWER_CHARS, HISTORY_TURNS } from "@/lib/api"
import { numberCitations } from "@/lib/citations"
import { emojiKey, splitEmoji } from "@/lib/emoji"
import { parsePassage, passagePreview } from "@/lib/passage"
import { numericColumns } from "@/lib/tables"
import { describe, phaseOf, sortDocuments } from "@/lib/status"
import { answer, doc } from "./helpers"

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } })

group("api client", () => {
  it("sends the question and its earlier exchanges to /api/chat", async () => {
    // a fresh Response per call: a body can be read only once
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(async () => json(answer()))
    await api.chat("What was revenue?")
    const [url, init] = fetch.mock.calls[0]
    expect(url).toBe("/api/chat")
    expect(init?.method).toBe("POST")
    expect(JSON.parse(init?.body as string)).toEqual({ question: "What was revenue?", history: [] })
    const history = [{ question: "What is this about?", answer: "An annual report [a.pdf, Page 1]." }]
    await api.chat("tell me more", history)
    expect(JSON.parse(fetch.mock.calls[1][1]?.body as string)).toEqual({ question: "tell me more", history })
  })

  it("keeps the server detail but maps status codes to user messages", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(json({ detail: "Vector database unavailable: ConnectError" }, 503))
    const error = await api.documents().catch((e) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error.status).toBe(503)
    expect(errorMessage(error, "load")).toBe("Document search is temporarily unavailable.")
    expect(errorMessage(error, "load")).not.toContain("ConnectError")
  })

  it("treats a network failure or the proxy's unreachable marker as status 0", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new TypeError("Failed to fetch"))
    expect((await api.documents().catch((e) => e)).status).toBe(0)
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(json({ detail: "x" }, 502, { "x-rag-unreachable": "1" }))
    const error = await api.chat("q").catch((e) => e)
    expect(error.status).toBe(0)
    expect(errorMessage(error, "chat")).toBe("Document service is unavailable.")
  })

  it("accepts a 503 health report as data, not an error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(json({ status: "unavailable", qdrant: false, collection: "documents" }, 503))
    expect(await api.health()).toMatchObject({ status: "unavailable" })
  })

  it.each([
    [413, "upload", "This PDF is larger than the 200 MB limit."],
    [415, "upload", "Only PDF documents are supported."],
    [409, "upload", "This document is already being processed."],
    [429, "upload", "Processing is busy. Try again in a moment."],
    [502, "chat", "The answer service encountered an error. Try again."],
    [503, "chat", "Document search is temporarily unavailable."],
    [500, "delete", "Couldn't delete the document. Try again."],
  ] as const)("maps %i during %s", (status, action, message) => {
    expect(errorMessage(new ApiError(status, "internal detail"), action)).toBe(message)
  })
})

group("citations", () => {
  it("numbers each cited page once and groups its passages", () => {
    const { markdown, groups } = numberCitations(answer())
    expect(markdown).toBe("Revenue reached 120 in 2025 [1](#cite-1), up from 100 [1](#cite-1).")
    expect(groups).toHaveLength(1)
    expect(groups[0]).toMatchObject({ n: 1, document: "Annual Report.pdf", page: 4 })
    expect(groups[0].passages).toHaveLength(2)
  })

  it("numbers pages from different documents in first-use order", () => {
    const { markdown, groups } = numberCitations({
      answer: "A [b.pdf, Page 2]. B [a.pdf, Page 9]. C [b.pdf, Page 2].",
      citations: [{ document: "b.pdf", page: 2 }, { document: "a.pdf", page: 9 }],
      sources: [],
    })
    expect(markdown).toBe("A [1](#cite-1). B [2](#cite-2). C [1](#cite-1).")
    expect(groups.map((g) => [g.n, g.document, g.page])).toEqual([[1, "b.pdf", 2], [2, "a.pdf", 9]])
  })
})

group("document status", () => {
  it("maps backend stages to the four user-facing phases", () => {
    expect(phaseOf(doc({ status: "queued", ingestion: { stage: "queued", pages: 0, chunks: 0 } }))).toBe(-1)
    expect(describe(doc({ status: "queued" }))).toBe("Waiting to process")
    expect(describe(doc({ status: "processing", ingestion: { stage: "extract", pages: 0, chunks: 0 } }))).toBe("Reading document")
    expect(describe(doc({ status: "processing", ingestion: { stage: "understand", pages: 9, chunks: 0 } }))).toBe("Understanding complex pages")
    expect(describe(doc({ status: "processing", ingestion: { stage: "embed", pages: 9, chunks: 40 } }))).toBe("Building index")
    expect(describe(doc({ status: "processing", ingestion: { stage: "finish", pages: 9, chunks: 40 } }))).toBe("Finalizing")
  })

  it("summarises stored documents", () => {
    expect(describe(doc({ pages_with_chunks: 1, chunks: 1 }))).toBe("1 indexed page · 1 passage")
    expect(describe(doc({ status: "failed" }))).toBe("Processing failed")
  })
})

group("emoji", () => {
  it("finds emoji (with skin tones, ZWJ sequences, flags, keycaps) and leaves text symbols alone", () => {
    const pieces = splitEmoji("Hi 👋🏽 team ❤️ © 2025 ✓ 🇮🇳 1️⃣ 👨‍💻!")
    expect(pieces.filter((p) => "emoji" in p).map((p) => (p as { emoji: string }).emoji)).toEqual(["👋🏽", "❤️", "🇮🇳", "1️⃣", "👨‍💻"])
    expect(pieces.map((p) => ("emoji" in p ? "·" : p.text)).join("")).toBe("Hi · team · © 2025 ✓ · · ·!")
  })

  it("maps an emoji to its Apple image name without U+FE0F", () => {
    expect(emojiKey("❤️")).toBe("2764")
    expect(emojiKey("👋🏽")).toBe("1f44b-1f3fd")
    expect(emojiKey("👨‍💻")).toBe("1f468-200d-1f4bb")
  })
})

group("passages", () => {
  it("turns pipe tables into header and rows, keeping cells as plain text", () => {
    const blocks = parsePassage("Revenue by year\n|Year|Revenue|\n|---|---|\n|2024|100|\n|2025|1<br>20 \\| est.|\nSource: audit")
    expect(blocks).toEqual([
      { type: "text", text: "Revenue by year" },
      { type: "table", header: ["Year", "Revenue"], rows: [["2024", "100"], ["2025", "1\n20 | est."]] },
      { type: "text", text: "Source: audit" },
    ])
  })

  it("keeps header-less tables and single pipe lines", () => {
    expect(parsePassage("|a|b|\n|c|d|")).toEqual([{ type: "table", header: null, rows: [["a", "b"], ["c", "d"]] }])
    expect(parsePassage("| just one line |")).toEqual([{ type: "text", text: "| just one line |" }])
  })

  it("previews tables as readable text", () => {
    expect(passagePreview("|Year|Revenue|\n|---|---|\n|2025|120|")).toBe("Table: Year, Revenue · 1 row")
    expect(passagePreview("Revenue by year\n|a|b|\n|c|d|")).toBe("Revenue by year Table · 2 rows")
    expect(passagePreview("x".repeat(300), 20)).toHaveLength(20)
  })
})

it("sorts documents: in progress, then needing attention, then ready", () => {
  const names = sortDocuments([
    doc({ source_name: "b.pdf" }),
    doc({ source_name: "a.pdf" }),
    doc({ source_name: "f.pdf", status: "failed" }),
    doc({ source_name: "p.pdf", status: "processing" }),
  ]).map((d) => d.source_name)
  expect(names).toEqual(["p.pdf", "f.pdf", "a.pdf", "b.pdf"])
})

it("right-aligns numeric columns, never the row-label column", () => {
  expect(numericColumns([["2024", "₹1,200 cr", "12.5%", "n/a"], ["2025", "(300)", "—", "7"]])).toEqual([false, true, true, false])
  expect(numericColumns([["Revenue", ""], ["Profit", "-"]])).toEqual([false, false])
})

group("chat history sent with a question", () => {
  const done = (question: string, response = answer()): Message => ({ id: question, role: "assistant", question, state: "done", response })

  it("keeps the last answered exchanges, without small talk, errors or pending replies", () => {
    const messages: Message[] = [
      { id: "u0", role: "user", text: "hi" },
      done("hi", answer({ answer: "Hey", kind: "conversation", citations: [], sources: [] })),
      ...Array.from({ length: HISTORY_TURNS + 1 }, (_, i) => done(`question ${i}`)),
      { id: "e", role: "assistant", question: "failed", state: "error", error: "Something went wrong." },
      done("not found", answer({ answer: "I couldn't find enough.", abstained: true, kind: "abstain", citations: [], sources: [] })),
      { id: "p", role: "assistant", question: "pending", state: "pending" },
    ]
    const history = chatHistory(messages)
    expect(history.map((t) => t.question)).toEqual(["question 2", "question 3", "not found"]) // abstentions still give context
    expect(history).toHaveLength(HISTORY_TURNS)
  })

  it("cuts long answers to the size the API accepts", () => {
    const [turn] = chatHistory([done("long", answer({ answer: "x".repeat(HISTORY_ANSWER_CHARS + 900) }))])
    expect(turn.answer).toHaveLength(HISTORY_ANSWER_CHARS)
  })
})
