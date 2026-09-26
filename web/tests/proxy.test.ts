import { expect, it, vi } from "vitest"

import { DELETE, GET, POST } from "@/app/api/[...path]/route"

const context = (path: string) => ({ params: Promise.resolve({ path: path.split("/") }) })

it("forwards only the endpoints the UI uses", async () => {
  const fetch = vi.spyOn(globalThis, "fetch").mockResolvedValue(Response.json({ documents: [] }))

  const ok = await GET(new Request("http://ui/api/documents"), context("documents"))
  expect(ok.status).toBe(200)
  expect(fetch.mock.calls[0][0]).toBe("http://127.0.0.1:8000/api/documents")

  for (const [handler, method, path] of [
    [GET, "GET", "docs"],
    [GET, "GET", "openapi.json"],
    [GET, "GET", "chat"],
    [POST, "POST", "health"],
    [POST, "POST", "documents/0123456789abcdef"],
    [DELETE, "DELETE", "documents"],
    [DELETE, "DELETE", "documents/../../admin"],
    [DELETE, "DELETE", "documents/not-a-document-id"],
    [DELETE, "DELETE", "documents/0123456789abcdef/x"],
  ] as const) {
    const response = await handler(new Request(`http://ui/api/${path}`, { method }), context(path))
    expect(response.status, `${method} ${path}`).toBe(404)
  }
  expect(fetch).toHaveBeenCalledTimes(1)
})

it("streams an upload to FastAPI without reading it, keeping the multipart content type", async () => {
  const fetch = vi.spyOn(globalThis, "fetch").mockResolvedValue(Response.json({ status: "queued" }, { status: 202 }))
  let pulled = 0
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      pulled += 1 // only the consumer (FastAPI's side) may pull from the stream
      controller.enqueue(new TextEncoder().encode("--b\r\n"))
      controller.close()
    },
  }, { highWaterMark: 0 }) // no eager pull: any read must come from the proxy
  const contentType = "multipart/form-data; boundary=----WebKitFormBoundaryABC123"
  const request = new Request("http://ui/api/documents", {
    method: "POST",
    body,
    headers: { "content-type": contentType, "content-length": "5", cookie: "session=secret", authorization: "Basic dXNlcjpwdw==" },
    // @ts-expect-error: Node's Request needs duplex for a stream body
    duplex: "half",
  })
  const response = await POST(request, context("documents"))

  expect(response.status).toBe(202)
  const [url, init] = fetch.mock.calls[0]
  expect(url).toBe("http://127.0.0.1:8000/api/documents")
  expect(init?.method).toBe("POST")
  expect(init?.body).toBe(request.body) // the same stream object: never buffered by the proxy
  expect(pulled).toBe(0)
  const headers = new Headers(init?.headers)
  expect(headers.get("content-type")).toBe(contentType) // boundary preserved
  expect(headers.get("content-length")).toBe("5")
  expect(headers.get("cookie")).toBeNull()
  expect(headers.get("authorization")).toBeNull()
})

it("forwards DELETE with its document id and no body", async () => {
  const fetch = vi.spyOn(globalThis, "fetch").mockResolvedValue(Response.json({ document_id: "0123456789abcdef", deleted_points: 4 }))
  const response = await DELETE(new Request("http://ui/api/documents/0123456789abcdef", { method: "DELETE" }), context("documents/0123456789abcdef"))
  expect(response.status).toBe(200)
  const [url, init] = fetch.mock.calls[0]
  expect(url).toBe("http://127.0.0.1:8000/api/documents/0123456789abcdef")
  expect(init?.method).toBe("DELETE")
  expect(init?.body).toBeNull()
})

it.each([
  [413, "File is larger than the 200 MB upload limit."],
  [415, "Only PDF files are accepted."],
  [409, "This document is already queued or being processed."],
  [429, "The ingestion queue is full. Try again when the current upload has finished."],
  [404, "No such document."],
])("passes a %i and its user-facing detail through", async (status, detail) => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(Response.json({ detail }, { status }))
  const response = await POST(new Request("http://ui/api/documents", { method: "POST", body: "x" }), context("documents"))
  expect(response.status).toBe(status)
  expect(await response.json()).toEqual({ detail })
})

it.each([503, 502, 500])("keeps a %i status but hides backend details", async (status) => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    Response.json({ detail: "Vector database unavailable: ResponseHandlingException: [Errno 111] qdrant:6333" }, { status }),
  )
  const response = await POST(new Request("http://ui/api/chat", { method: "POST", body: "{}" }), context("chat"))
  expect(response.status).toBe(status)
  const text = await response.text()
  expect(text).not.toMatch(/qdrant|Vector database|Exception/i)
  expect(response.headers.get("x-rag-unreachable")).toBeNull()
})

it("marks an unreachable API", async () => {
  vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new TypeError("fetch failed"))
  const down = await GET(new Request("http://ui/api/health"), context("health"))
  expect(down.status).toBe(502)
  expect(down.headers.get("x-rag-unreachable")).toBe("1")
})

it("reduces replies to the fields the UI uses", async () => {
  const fetch = vi.spyOn(globalThis, "fetch")
  fetch.mockResolvedValueOnce(Response.json({
    answer: "Revenue was 120 [a.pdf, Page 1].", abstained: false,
    citations: [{ document: "a.pdf", document_id: "0123456789abcdef", page: 1 }],
    sources: [{ chunk_id: "0123456789abcdef-p1-0", document_id: "0123456789abcdef", source_name: "a.pdf", page_number: 1, section_path: ["Financials"], content_type: "table", text: "|2025|120|" }],
  }))
  const chat = await (await POST(new Request("http://ui/api/chat", { method: "POST", body: "{}" }), context("chat"))).json()
  expect(chat).toEqual({
    answer: "Revenue was 120 [a.pdf, Page 1].", abstained: false,
    citations: [{ document: "a.pdf", page: 1 }],
    sources: [{ source_name: "a.pdf", page_number: 1, section_path: ["Financials"], content_type: "table", text: "|2025|120|" }],
  })

  fetch.mockResolvedValueOnce(Response.json({ documents: [{
    document_id: "0123456789abcdef", source_name: "a.pdf", status: "failed", chunks: 0, pages_with_chunks: 0, content_types: { text: 3 },
    ingestion: { stage: "embed", pages: 3, chunks: 3, new_embeddings: 2, error: "ClientError" },
  }] }))
  const docs = await (await GET(new Request("http://ui/api/documents"), context("documents"))).json()
  expect(docs).toEqual({ documents: [{
    document_id: "0123456789abcdef", source_name: "a.pdf", status: "failed", chunks: 0, pages_with_chunks: 0,
    ingestion: { stage: "embed", pages: 3, chunks: 3 },
  }] })

  fetch.mockResolvedValueOnce(Response.json({ status: "unavailable", qdrant: false, collection: "documents" }, { status: 503 }))
  const health = await GET(new Request("http://ui/api/health"), context("health"))
  expect(health.status).toBe(503)
  expect(await health.json()).toEqual({ status: "unavailable" })

  fetch.mockResolvedValueOnce(Response.json({ document_id: "0123456789abcdef", deleted_points: 4 }))
  const deleted = await DELETE(new Request("http://ui/api/documents/0123456789abcdef", { method: "DELETE" }), context("documents/0123456789abcdef"))
  expect(await deleted.json()).toEqual({ deleted: true })

  fetch.mockResolvedValueOnce(Response.json({ detail: [{ type: "string_too_short", loc: ["body", "question"], input: "" }] }, { status: 422 }))
  const invalid = await POST(new Request("http://ui/api/chat", { method: "POST", body: "{}" }), context("chat"))
  expect(invalid.status).toBe(422)
  expect(await invalid.json()).toEqual({ detail: "Request failed." })
})
