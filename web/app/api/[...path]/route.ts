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

/* eslint-disable @typescript-eslint/no-explicit-any -- upstream JSON, reduced field by field */
const pick = (o: any, keys: string[]) => Object.fromEntries(keys.filter((k) => o?.[k] !== undefined).map((k) => [k, o[k]]))

const PROJECT: Record<string, (body: any) => unknown> = {
  "GET health": (b) => pick(b, ["status"]),
  "GET documents": (b) => ({
    documents: (b.documents ?? []).map((d: any) => ({
      ...pick(d, ["document_id", "source_name", "status", "chunks", "pages_with_chunks"]),
      ingestion: d.ingestion ? pick(d.ingestion, ["stage", "pages", "chunks"]) : null,
    })),
  }),
  "POST documents": (b) => pick(b, ["source_name", "status"]),
  "POST chat": (b) => ({
    ...pick(b, ["answer", "abstained"]),
    citations: (b.citations ?? []).map((c: any) => pick(c, ["document", "page"])),
    sources: (b.sources ?? []).map((s: any) => pick(s, ["source_name", "page_number", "section_path", "content_type", "text"])),
  }),
  "DELETE documents": () => ({ deleted: true }),
}
/* eslint-enable @typescript-eslint/no-explicit-any */

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
