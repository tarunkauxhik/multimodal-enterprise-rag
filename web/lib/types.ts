// Response shapes of the FastAPI app (api.py). Only the fields the UI uses are declared;
// extra fields in a response are ignored.

export type DocumentStatus = "ready" | "incomplete" | "queued" | "processing" | "failed" | "empty"

export interface IngestionStatus {
  stage: string
  pages: number
  chunks: number
}

export interface DocumentSummary {
  document_id: string // used only as the DELETE key, never shown
  source_name: string
  status: DocumentStatus
  chunks: number
  pages_with_chunks: number
  ingestion?: IngestionStatus | null
}

export interface Health {
  status: "ok" | "unavailable"
}

export interface IngestAccepted {
  source_name: string
  status: "queued" | "ready"
}

export interface Citation {
  document: string
  page: number
}

export interface Source {
  source_name: string
  page_number: number
  section_path: string[]
  content_type: string
  text: string
}

// answer: grounded and cited. abstain: not enough evidence. out_of_scope: not about the documents.
// conversation: small talk. workspace: from document metadata. clarify: a question back to the user.
export type ReplyKind = "answer" | "abstain" | "out_of_scope" | "conversation" | "workspace" | "clarify"

export interface ChatResponse {
  answer: string
  abstained: boolean
  citations: Citation[]
  sources: Source[]
  kind?: ReplyKind // absent in conversations saved before the field existed
  suggestions?: string[] // one-tap follow-up questions
}

/** The reply kind, also for replies saved before `kind` existed. */
export const replyKind = (r: ChatResponse): ReplyKind => r.kind ?? (r.abstained ? "abstain" : "answer")
