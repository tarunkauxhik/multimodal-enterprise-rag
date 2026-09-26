// Server-side proxy: the browser only ever talks to this Next.js server; FastAPI stays on
// 127.0.0.1 (RAG_API_URL). Only the five endpoints the UI uses are forwarded.
//
// Requests: bodies are streamed (request.body is passed on, never read), so a 200 MB upload is
// never buffered here. Only content-type/length and accept are forwarded (never cookies or the
// Authorization header from Nginx basic auth). Nothing is logged.
//
// Responses: this is the public boundary, so each JSON reply is reduced to the fields the UI uses
// (no chunk ids, point counts, exception names or database details), and error details pass
// through only for the fixed, user-safe 4xx messages.

const API_URL = (process.env.RAG_API_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "")

const ROUTES: Record<string, RegExp> = {
  GET: /^(health|documents)$/,
  POST: /^(documents|chat)$/,
  DELETE: /^documents\/[0-9a-f]{16}$/,
}

const SAFE_DETAIL = new Set([404, 409, 413, 415, 429]) // fixed strings in api.py, no internals
const KINDS = new Set(["answer", "abstain", "out_of_scope", "conversation", "workspace", "clarify"])

const DOCUMENT_STATUS = new Set(["ready", "incomplete", "queued", "processing", "failed", "empty"])
const CONTENT_TYPES = new Set(["text", "table", "figure"])

// Runtime validation: TypeScript types say nothing about what the upstream actually sends. Every
// value is type-checked; anything unexpected becomes a safe default or its entry is dropped.
type Json = Record<string, unknown>
const obj = (v: unknown): Json => (v && typeof v === "object" && !Array.isArray(v) ? (v as Json) : {})
const list = (v: unknown): unknown[] => (Array.isArray(v) ? v : [])
const str = (v: unknown, max = 200_000): string => (typeof v === "string" ? v.slice(0, max) : "")
const count = (v: unknown): number => (Number.isInteger(v) && (v as number) >= 0 ? (v as number) : 0)
const page = (v: unknown): number | null => (Number.isInteger(v) && (v as number) >= 1 ? (v as number) : null)
const oneOf = (v: unknown, allowed: Set<string>, fallback: string): string => (typeof v === "string" && allowed.has(v) ? v : fallback)

const PROJECT: Record<string, (body: unknown) => unknown> = {
  "GET health": (b) => ({ status: obj(b).status === "ok" ? "ok" : "unavailable" }),
  "GET documents": (b) => ({
    documents: list(obj(b).documents)
      .map(obj)
      .filter((d) => typeof d.document_id === "string" && /^[0-9a-f]{16}$/.test(d.document_id) && str(d.source_name))
      .map((d) => ({
        document_id: d.document_id, // the DELETE key; never displayed
        source_name: str(d.source_name, 500),
        status: oneOf(d.status, DOCUMENT_STATUS, "failed"),
        chunks: count(d.chunks),
        pages_with_chunks: count(d.pages_with_chunks),
        ingestion: d.ingestion ? { stage: str(obj(d.ingestion).stage, 40), pages: count(obj(d.ingestion).pages), chunks: count(obj(d.ingestion).chunks) } : null,
      })),
  }),
  "POST documents": (b) => ({ source_name: str(obj(b).source_name, 500), status: oneOf(obj(b).status, new Set(["queued", "ready"]), "queued") }),
  "POST chat": (b) => {
    const body = obj(b)
    const abstained = body.abstained === true
    return {
      answer: str(body.answer),
      abstained,
      kind: oneOf(body.kind, KINDS, abstained ? "abstain" : "answer"),
      suggestions: list(body.suggestions).filter((s): s is string => typeof s === "string" && s.length > 0 && s.length <= 200).slice(0, 3),
      citations: list(body.citations)
        .map(obj)
        .map((c) => ({ document: str(c.document, 500), page: page(c.page) }))
        .filter((c) => c.document && c.page !== null),
      sources: list(body.sources)
        .map(obj)
        .map((s) => ({
          source_name: str(s.source_name, 500),
          page_number: page(s.page_number),
          section_path: list(s.section_path).filter((x): x is string => typeof x === "string").slice(0, 12).map((x) => x.slice(0, 300)),
          content_type: oneOf(s.content_type, CONTENT_TYPES, "text"),
          text: str(s.text),
        }))
        .filter((s) => s.source_name && s.page_number !== null),
    }
  },
  "DELETE documents": () => ({ deleted: true }),
}

const json = (body: unknown, status: number, headers: Record<string, string> = {}) =>
  Response.json(body, { status, headers: { "cache-control": "no-store", ...headers } })

async function forward(request: Request, context: { params: Promise<{ path: string[] }> }) {
  const path = (await context.params).path.join("/")
  if (!ROUTES[request.method]?.test(path)) return json({ detail: "Not found" }, 404)

  const headers = new Headers()
  for (const name of ["content-type", "content-length", "accept"]) {
    const value = request.headers.get(name)
    if (value) headers.set(name, value)
  }
  let upstream: Response
  try {
    upstream = await fetch(`${API_URL}/api/${path}`, {
      method: request.method,
      headers,
      body: request.body,
      cache: "no-store",
      // @ts-expect-error: required by Node's fetch for a streamed request body; not in the DOM types
      duplex: "half",
    })
  } catch {
    return json({ detail: "Document service is unavailable." }, 502, { "x-rag-unreachable": "1" })
  }

  let body: unknown = null
  try {
    body = await upstream.json() // small JSON replies only; uploads flow the other way
  } catch {}
  const ok = upstream.ok || (path === "health" && upstream.status === 503) // health reports an outage as data
  if (!ok) {
    const detail = SAFE_DETAIL.has(upstream.status) ? (body as { detail?: unknown })?.detail : undefined
    return json({ detail: typeof detail === "string" ? detail : "Request failed." }, upstream.status)
  }
  return json(PROJECT[`${request.method} ${path.split("/")[0]}`](body ?? {}), upstream.status)
}

export { forward as GET, forward as POST, forward as DELETE }
