import { act, fireEvent, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { expect, it, vi } from "vitest"

import { DocumentsView } from "@/components/documents/documents-view"
import { validateFile } from "@/hooks/use-documents"
import { ApiError } from "@/lib/api"
import { doc, mockApi, renderWorkspace } from "./helpers"

const pdf = (name = "new.pdf", size = 1024) => {
  const file = new File(["%PDF-1.7"], name, { type: "application/pdf" })
  Object.defineProperty(file, "size", { value: size })
  return file
}

it("labels every document status with text, not color alone", async () => {
  mockApi([
    doc({ document_id: "a".repeat(16), source_name: "queued.pdf", status: "queued", ingestion: { stage: "queued", pages: 0, chunks: 0 } }),
    doc({ document_id: "b".repeat(16), source_name: "processing.pdf", status: "processing", ingestion: { stage: "understand", pages: 9, chunks: 0 } }),
    doc({ document_id: "c".repeat(16), source_name: "failed.pdf", status: "failed" }),
    doc({ document_id: "d".repeat(16), source_name: "incomplete.pdf", status: "incomplete" }),
    doc({ document_id: "e".repeat(16), source_name: "empty.pdf", status: "empty" }),
    doc({ document_id: "f".repeat(16), source_name: "ready.pdf" }),
  ])
  renderWorkspace(<DocumentsView />)
  const list = await screen.findByRole("region", { name: "All documents" })
  await within(list).findByText("ready.pdf")
  for (const text of [
    "Waiting to process",
    "Processing document",
    "Understanding complex pages",
    "Processing failed",
    "Index incomplete",
    "The document was only partially indexed. Upload the same file again to repair it.",
    "No readable content",
    "12 pages · 42 passages",
  ]) {
    expect(within(list).getAllByText(text).length, text).toBeGreaterThan(0)
  }
  expect(within(list).getByRole("progressbar", { name: "Step 2 of 4: Understanding complex pages" })).toBeInTheDocument()
  expect(screen.getByText("1 of 6 ready")).toBeInTheDocument()
})

it("polls only while a document is processing", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  const api = mockApi([doc({ status: "processing", ingestion: { stage: "embed", pages: 3, chunks: 9 } })])
  renderWorkspace(<DocumentsView />)
  await screen.findAllByText("Building index")
  api.documents.mockResolvedValue([doc()])
  await act(() => vi.advanceTimersByTimeAsync(2100))
  await screen.findAllByText("12 pages · 42 passages")
  const calls = api.documents.mock.calls.length
  await act(() => vi.advanceTimersByTimeAsync(10_000))
  expect(api.documents.mock.calls.length).toBe(calls) // nothing active: no more polling
  vi.useRealTimers()
})

it("uploads PDFs from the picker and reports server rejections", async () => {
  const api = mockApi([])
  api.upload.mockRejectedValueOnce(new ApiError(413))
  const { container } = renderWorkspace(<DocumentsView />)
  await screen.findByText("No documents yet", { selector: "p.text-sm" })
  const input = container.ownerDocument.querySelector<HTMLInputElement>("input[type=file]")!
  fireEvent.change(input, { target: { files: [pdf("big.pdf")] } })
  expect(await screen.findByText("big.pdf: This PDF is larger than the 200 MB limit.")).toBeInTheDocument()

  api.documents.mockResolvedValue([doc({ source_name: "new.pdf", status: "queued" })])
  fireEvent.change(input, { target: { files: [pdf("new.pdf")] } })
  await waitFor(() => expect(api.upload).toHaveBeenCalledTimes(2))
  expect(await screen.findAllByText("Waiting to process")).not.toHaveLength(0)
})

it("rejects non-PDF and oversized files before uploading", () => {
  expect(validateFile(new File(["x"], "notes.txt", { type: "text/plain" }))).toBe("Only PDF documents are supported.")
  expect(validateFile(pdf("huge.pdf", 201 * 1024 * 1024))).toBe("This PDF is larger than the 200 MB limit.")
  expect(validateFile(pdf())).toBeNull()
})

it("deletes a document after confirmation", async () => {
  const api = mockApi([doc()])
  renderWorkspace(<DocumentsView />)
  const user = userEvent.setup()
  await user.click(await screen.findByRole("button", { name: "Actions for Annual Report.pdf" }))
  await user.click(await screen.findByRole("menuitem", { name: "Delete" }))

  const dialog = await screen.findByRole("alertdialog", { name: "Delete this document?" })
  expect(dialog).toHaveTextContent("removed from the shared workspace")
  api.documents.mockResolvedValue([])
  await user.click(within(dialog).getByRole("button", { name: "Delete" }))

  expect(api.deleteDocument).toHaveBeenCalledWith("0123456789abcdef")
  expect(await screen.findByText("Annual Report.pdf was deleted.")).toBeInTheDocument()
  await waitFor(() => expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument())
})

it("keeps the document and explains when deletion fails", async () => {
  const api = mockApi([doc()])
  api.deleteDocument.mockRejectedValue(new ApiError(503))
  renderWorkspace(<DocumentsView />)
  const user = userEvent.setup()
  await user.click(await screen.findByRole("button", { name: "Actions for Annual Report.pdf" }))
  await user.click(await screen.findByRole("menuitem", { name: "Delete" }))
  await user.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Delete" }))
  expect(await screen.findByText("Document search is temporarily unavailable.")).toBeInTheDocument()
})

it("does not offer deletion while a document is processing", async () => {
  mockApi([doc({ status: "processing", ingestion: { stage: "extract", pages: 0, chunks: 0 } })])
  renderWorkspace(<DocumentsView />)
  const user = userEvent.setup()
  await user.click(await screen.findByRole("button", { name: "Actions for Annual Report.pdf" }))
  expect(await screen.findByRole("menuitem", { name: "Delete (after processing)" })).toHaveAttribute("aria-disabled", "true")
})

it("toggles the sidebar with Ctrl+B", async () => {
  mockApi()
  const { container } = renderWorkspace(<DocumentsView />)
  const sidebar = () => container.querySelector("[data-slot=sidebar][data-state]")!
  await screen.findAllByText("Annual Report.pdf")
  expect(sidebar()).toHaveAttribute("data-state", "expanded")
  await userEvent.setup().keyboard("{Control>}b{/Control}")
  expect(sidebar()).toHaveAttribute("data-state", "collapsed")
})

it("announces when a processing document becomes ready", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  const api = mockApi([doc({ status: "processing", ingestion: { stage: "embed", pages: 3, chunks: 9 } })])
  renderWorkspace(<DocumentsView />)
  await screen.findAllByText("Building index")
  api.documents.mockResolvedValue([doc()])
  await act(() => vi.advanceTimersByTimeAsync(2100))
  expect(await screen.findByText("Annual Report.pdf is ready for questions.")).toBeInTheDocument()
  vi.useRealTimers()
})
