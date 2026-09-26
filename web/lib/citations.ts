// Presentation only: the backend has already validated every citation and normalised it in the
// answer text to "[document, Page N]". Here each (document, page) gets a number, the answer is
// rewritten to link to it, and the passages behind it are grouped for display.

import type { ChatResponse, Source } from "@/lib/types"

export interface SourceGroup {
  n: number
  document: string
  page: number
  passages: Source[] // one or more retrieved passages from this document page
}

export const CITE_PREFIX = "#cite-"

export function numberCitations(response: Pick<ChatResponse, "answer" | "citations" | "sources">) {
  const groups: SourceGroup[] = []
  let markdown = response.answer
  for (const citation of response.citations) {
    const n = groups.length + 1
    groups.push({
      n,
      document: citation.document,
      page: citation.page,
      passages: response.sources.filter((s) => s.source_name === citation.document && s.page_number === citation.page),
    })
    markdown = markdown.split(`[${citation.document}, Page ${citation.page}]`).join(`\u0000${n}\u0000`)
  }
  // Markdown formatting from the model (lists, bold, tables) is kept; links, images and HTML are not
  // rendered by AnswerMarkdown. Citation markers are placed as #cite-N links.
  markdown = markdown.replace(/\u0000(\d+)\u0000/g, (_, n) => `[${n}](${CITE_PREFIX}${n})`)
  return { markdown, groups }
}

export const CONTENT_TYPE: Record<string, string> = { text: "Text", table: "Table", figure: "Figure" }
export const CONTENT_EMOJI: Record<string, string> = { text: "📄", table: "📊", figure: "🖼️" }
