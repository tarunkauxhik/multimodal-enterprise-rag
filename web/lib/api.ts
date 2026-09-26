// The only place the UI talks to the backend. Requests go to same-origin /api/*, which the
// Next.js server forwards to FastAPI (app/api/[...path]/route.ts).

import type { ChatResponse, DocumentSummary, Health, IngestAccepted } from "@/lib/types"

export const MAX_UPLOAD_MB = 200 // matches the API default (API_MAX_UPLOAD_MB)
export const MAX_QUESTION_CHARS = 2000 // matches ChatRequest in api.py
export const UNREACHABLE_HEADER = "x-rag-unreachable" // set by the proxy when FastAPI does not answer

/** status 0: the document service could not be reached at all. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string = "",
  ) {
    super(`API error ${status}`)
  }
}

type Action = "load" | "upload" | "delete" | "chat"

/** A short user-facing message. Server detail text is never shown: it can name internals. */
export function errorMessage(error: unknown, action: Action): string {
  const status = error instanceof ApiError ? error.status : -1
  switch (status) {
    case 0:
      return "Document service is unavailable."
    case 413:
      return `This PDF is larger than the ${MAX_UPLOAD_MB} MB limit.`
    case 415:
      return "Only PDF documents are supported."
    case 409:
      return action === "upload" ? "This document is already being processed." : "This document is busy. Try again once processing finishes."
    case 429:
      return "Processing is busy. Try again in a moment."
    case 404:
      return "This document no longer exists."
    case 503:
      return "Document search is temporarily unavailable."
    case 502:
      return "The answer service encountered an error. Try again."
    case 422:
      return "That request was not valid."
  }
  return {
    load: "Couldn't load your documents.",
    upload: "Upload failed. Try again.",
    delete: "Couldn't delete the document. Try again.",
    chat: "Something went wrong. Try again.",
  }[action]
}

async function failure(response: Response): Promise<ApiError> {
  if (response.headers.get(UNREACHABLE_HEADER)) return new ApiError(0)
  let detail = ""
  try {
    const body = await response.json()
    if (typeof body?.detail === "string") detail = body.detail
  } catch {}
  return new ApiError(response.status, detail)
}

async function request<T>(path: string, init?: RequestInit, ok: number[] = []): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, { cache: "no-store", ...init })
  } catch {
    throw new ApiError(0)
  }
  if (!response.ok && !ok.includes(response.status)) throw await failure(response)
  return (await response.json()) as T
}

export const api = {
  health: () => request<Health>("/api/health", undefined, [503]),

  documents: async () => (await request<{ documents: DocumentSummary[] }>("/api/documents")).documents,

  deleteDocument: (documentId: string) =>
    request<unknown>(`/api/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" }),

  chat: (question: string, signal?: AbortSignal) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question }), // single-turn: earlier messages are never sent
      signal,
    }),

  /** XMLHttpRequest rather than fetch: fetch cannot report upload progress. */
  upload: (file: File, onProgress?: (fraction: number) => void) =>
    new Promise<IngestAccepted>((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open("POST", "/api/documents")
      xhr.responseType = "json"
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) onProgress?.(event.loaded / event.total)
      }
      xhr.onerror = () => reject(new ApiError(0))
      xhr.onload = () => {
        if (xhr.status === 200 || xhr.status === 202) return resolve(xhr.response as IngestAccepted)
        if (xhr.getResponseHeader(UNREACHABLE_HEADER)) return reject(new ApiError(0))
        const detail = typeof xhr.response?.detail === "string" ? xhr.response.detail : ""
        reject(new ApiError(xhr.status, detail))
      }
      const form = new FormData()
      form.append("file", file, file.name)
      xhr.send(form)
    }),
}
