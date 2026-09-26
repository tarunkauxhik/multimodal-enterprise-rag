"use client"

import ReactMarkdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"

import { Emoji } from "@/components/emoji"
import { CITE_PREFIX } from "@/lib/citations"
import { rehypeAppleEmoji } from "@/lib/emoji"
import { rehypeNumericColumns } from "@/lib/tables"

/**
 * Renders the model's answer. Raw HTML is dropped (skipHtml), images are not rendered, and the only
 * links are citation markers (#cite-N); any other link keeps its text but loses its target, so
 * document content can never create a working link, image or element. GFM tables render as real
 * tables; emojis render as Apple emoji images.
 */
export function AnswerMarkdown({ markdown, renderCitation }: { markdown: string; renderCitation: (n: number) => React.ReactNode }) {
  const components: Components = {
    a: ({ href, children }) =>
      href?.startsWith(CITE_PREFIX) ? renderCitation(Number(href.slice(CITE_PREFIX.length))) : <span>{children}</span>,
    img: () => null,
    span: ({ node, children }) => {
      const emoji = node?.properties?.dataEmoji
      return typeof emoji === "string" ? <Emoji char={emoji} /> : <span>{children}</span>
    },
    table: ({ children }) => (
      <div className="data-table overflow-x-auto rounded-md border">
        <table>{children}</table>
      </div>
    ),
    th: ({ node, children }) => (
      <th scope="col" data-numeric={node?.properties?.dataNumeric !== undefined ? "" : undefined}>
        {children}
      </th>
    ),
    td: ({ node, children }) => <td data-numeric={node?.properties?.dataNumeric !== undefined ? "" : undefined}>{children}</td>,
    // A paragraph holding only citation markers (e.g. after a table or list) gets a quiet label.
    p: ({ node, children }) => {
      const parts = node?.children ?? []
      const markers = parts.filter((c) => c.type === "element" && c.tagName === "a" && String(c.properties?.href ?? "").startsWith(CITE_PREFIX))
      const onlyMarkers = markers.length > 0 && parts.every((c) => markers.includes(c) || (c.type === "text" && /^[\s.,;]*$/.test(c.value)))
      if (!onlyMarkers) return <p>{children}</p>
      return (
        <p className="citation-line text-sm text-muted-foreground">
          {markers.length === 1 ? "Source" : "Sources"} {children}
        </p>
      )
    },
  }
  return (
    <div className="answer">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeNumericColumns, rehypeAppleEmoji]} skipHtml components={components}>
        {markdown}
      </ReactMarkdown>
    </div>
  )
}
