import type { DocumentStatus, DocumentSummary } from "@/lib/types"

export type Tone = "neutral" | "active" | "success" | "warning" | "danger"

export const STATUS: Record<DocumentStatus, { label: string; tone: Tone; help?: string }> = {
  queued: { label: "Waiting to process", tone: "active" },
  processing: { label: "Processing document", tone: "active" },
  ready: { label: "Ready", tone: "success" },
  failed: {
    label: "Processing failed",
    tone: "danger",
    help: "This PDF couldn't be processed. Upload it again, or check that it opens correctly.",
  },
  incomplete: {
    label: "Index incomplete",
    tone: "warning",
    help: "The document was only partially indexed. Upload the same file again to repair it.",
  },
  empty: {
    label: "No readable content",
    tone: "neutral",
    help: "No text could be read from this PDF, so it can't be used for answers.",
  },
}

// Backend ingestion stages, grouped into the four steps users see.
export const PHASES = ["Reading document", "Understanding complex pages", "Building index", "Finalizing"] as const
const STAGE_PHASE: Record<string, number> = {
  setup: 0, extract: 0, understand: 1, chunk: 2, embed: 2, store: 2, finish: 3, done: 3,
}

/** Index into PHASES, or -1 while the document is still waiting. */
export function phaseOf(doc: DocumentSummary): number {
  if (doc.status !== "processing") return -1
  return STAGE_PHASE[doc.ingestion?.stage ?? ""] ?? 0
}

export const isActive = (doc: DocumentSummary) => doc.status === "queued" || doc.status === "processing"

export function describe(doc: DocumentSummary): string {
  const phase = phaseOf(doc)
  if (phase >= 0) return PHASES[phase]
  if (doc.status === "queued") return "Waiting to process"
  if (doc.status === "ready" || doc.status === "incomplete") {
    const pages = doc.pages_with_chunks
    return `${pages} ${pages === 1 ? "page" : "pages"} · ${doc.chunks} ${doc.chunks === 1 ? "passage" : "passages"}`
  }
  return STATUS[doc.status].label
}

// Order for lists: work in progress first, then documents needing attention, then ready; by name within.
const ORDER: Record<DocumentStatus, number> = { processing: 0, queued: 1, failed: 2, incomplete: 3, empty: 4, ready: 5 }

export function sortDocuments(docs: DocumentSummary[]): DocumentSummary[] {
  return [...docs].sort((a, b) => ORDER[a.status] - ORDER[b.status] || a.source_name.localeCompare(b.source_name))
}
