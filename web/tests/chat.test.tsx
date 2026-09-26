import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import axe from "axe-core"
import { expect, it } from "vitest"

import { ChatView } from "@/components/chat/chat-view"
import { ApiError } from "@/lib/api"
import { answer, doc, mockApi, renderWorkspace } from "./helpers"

async function ask(question: string) {
  const user = userEvent.setup()
  const box = await screen.findByRole("textbox", { name: "Question" })
  await waitFor(() => expect(box).toBeEnabled())
  await user.type(box, question)
  await user.keyboard("{Enter}")
  return user
}

it("shows a quiet empty state when there are no documents", async () => {
  mockApi([])
  renderWorkspace(<ChatView />)
  expect(await screen.findByRole("heading", { name: "Ask questions about your documents" })).toBeInTheDocument()
  expect(within(screen.getByRole("main")).getByRole("button", { name: "Add documents" })).toBeInTheDocument()
  expect(screen.getByRole("textbox", { name: "Question" })).toBeDisabled()
})

it("explains that questions wait for processing documents", async () => {
  mockApi([doc({ status: "processing", ingestion: { stage: "extract", pages: 0, chunks: 0 } })])
  renderWorkspace(<ChatView />)
  expect(await screen.findByRole("heading", { name: "Preparing your documents" })).toBeInTheDocument()
  expect(screen.getByRole("textbox", { name: "Question" })).toBeDisabled()
})

it("reports an unavailable document service", async () => {
  const api = mockApi()
  api.documents.mockRejectedValue(new ApiError(0))
  renderWorkspace(<ChatView />)
  expect(await screen.findByRole("heading", { name: "Document service is unavailable." })).toBeInTheDocument()
})

it("answers with numbered citations and grouped, plain-text sources", async () => {
  const api = mockApi()
  renderWorkspace(<ChatView />)
  const user = await ask("What was revenue?")

  expect(api.chat).toHaveBeenCalledWith("What was revenue?")
  const sources = await screen.findByRole("region", { name: "Sources" })
  // two passages from the same page are one source
  expect(within(sources).getAllByRole("button")).toHaveLength(1)
  expect(within(sources).getByRole("button")).toHaveTextContent("Annual Report.pdf")
  expect(within(sources).getByRole("button")).toHaveTextContent("p. 4")
  expect(screen.getAllByRole("button", { name: "Source 1: Annual Report.pdf, page 4" })).toHaveLength(2)
  expect(screen.queryByText(/\[Annual Report\.pdf, Page 4\]/)).not.toBeInTheDocument()

  await user.click(screen.getAllByRole("button", { name: "Source 1: Annual Report.pdf, page 4" })[0])
  const details = await screen.findByRole("dialog")
  expect(within(details).getByText("Revenue grew.")).toBeInTheDocument()
  expect(within(details).getByText(/Table · Financials/)).toBeInTheDocument()

  const results = await axe.run(document.body, { rules: { "color-contrast": { enabled: false } } })
  expect(results.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.html.slice(0, 120)).join(" | ")}`)).toEqual([])
})

it("never renders HTML, links or images from answers or documents", async () => {
  const api = mockApi()
  api.chat.mockResolvedValue(
    answer({
      answer: 'Click [here](https://evil.example) ![x](https://evil.example/x.png) <img src=x onerror="alert(1)"> [Annual Report.pdf, Page 4]',
      sources: [{ source_name: "Annual Report.pdf", page_number: 4, section_path: [], content_type: "text", text: '<script>alert(1)</script><b>bold</b>' }],
    }),
  )
  renderWorkspace(<ChatView />)
  const user = await ask("Anything?")
  await screen.findByRole("region", { name: "Sources" })

  const main = screen.getByRole("main")
  expect(main.querySelector("img, script, a[href^='http']")).toBeNull()
  expect(within(main).getByText(/Click/)).toHaveTextContent("Click here")

  await user.click(within(screen.getByRole("region", { name: "Sources" })).getByRole("button"))
  const details = await screen.findByRole("dialog")
  expect(within(details).getByText("<script>alert(1)</script><b>bold</b>")).toBeInTheDocument()
  expect(details.querySelector("script, b")).toBeNull()
})

it("shows an abstention as a neutral note without sources", async () => {
  const api = mockApi()
  api.chat.mockResolvedValue({ answer: "I could not find enough information.", abstained: true, citations: [], sources: [] })
  renderWorkspace(<ChatView />)
  await ask("Who is the CEO?")
  expect(await screen.findByText("I could not find enough information.")).toBeInTheDocument()
  expect(screen.queryByRole("region", { name: "Sources" })).not.toBeInTheDocument()
})

it("shows a chat error with a retry", async () => {
  const api = mockApi()
  api.chat.mockRejectedValueOnce(new ApiError(502, "Could not answer the question: RuntimeError"))
  renderWorkspace(<ChatView />)
  const user = await ask("What was revenue?")
  const alert = await screen.findByRole("alert")
  expect(alert).toHaveTextContent("The answer service encountered an error. Try again.")
  expect(alert).not.toHaveTextContent("RuntimeError")

  await user.click(within(alert).getByRole("button", { name: "Retry" }))
  expect(await screen.findByRole("region", { name: "Sources" })).toBeInTheDocument()
  expect(api.chat).toHaveBeenCalledTimes(2)
})

it("sends with Enter, adds a newline with Shift+Enter, and lists the conversation", async () => {
  const api = mockApi()
  renderWorkspace(<ChatView />)
  const user = userEvent.setup()
  const box = await screen.findByRole("textbox", { name: "Question" })
  await waitFor(() => expect(box).toBeEnabled())
  await user.type(box, "first line{Shift>}{Enter}{/Shift}second")
  expect(api.chat).not.toHaveBeenCalled()
  await user.keyboard("{Enter}")
  expect(api.chat).toHaveBeenCalledWith("first line\nsecond")
  expect(box).toHaveValue("")
  expect(await screen.findByRole("button", { name: /^first line/ })).toBeInTheDocument() // sidebar history
})

it("renders tables in answers and sources as real tables, and emojis as Apple images", async () => {
  const api = mockApi()
  api.chat.mockResolvedValue(
    answer({
      answer: "Results 👋\n\n| Year | Revenue |\n|---|---|\n| 2025 | 120 |\n\n[Annual Report.pdf, Page 4]",
      sources: [{ source_name: "Annual Report.pdf", page_number: 4, section_path: ["Financials"], content_type: "table", text: "|Year|Revenue|\n|---|---|\n|2025|120|" }],
    }),
  )
  renderWorkspace(<ChatView />)
  const user = await ask("Revenue table?")
  await screen.findByRole("region", { name: "Sources" })

  const main = screen.getByRole("main")
  const table = within(main).getByRole("table")
  expect(within(table).getAllByRole("columnheader").map((th) => th.textContent)).toEqual(["Year", "Revenue"])
  expect(within(table).getByRole("cell", { name: "120" })).toBeInTheDocument()
  expect(main).not.toHaveTextContent("|---|")
  expect(within(table).getByRole("cell", { name: "120" })).toHaveAttribute("data-numeric")
  expect(within(table).getByRole("cell", { name: "2025" })).not.toHaveAttribute("data-numeric")
  expect(main.querySelector(".citation-line")).toHaveTextContent("Source 1") // a citation alone after the table gets a label
  const wave = within(main).getByRole("img", { name: "👋" })
  expect(wave).toHaveAttribute("src", "/emoji/1f44b.png")

  await user.click(within(screen.getByRole("region", { name: "Sources" })).getByRole("button"))
  const details = await screen.findByRole("dialog")
  expect(within(details).getAllByRole("columnheader").map((th) => th.textContent)).toEqual(["Year", "Revenue"])
  expect(details).not.toHaveTextContent("|2025|")
})

it("offers starter questions about the ready documents", async () => {
  const api = mockApi([doc({ source_name: "Travel Policy.pdf" })])
  renderWorkspace(<ChatView />)
  const suggestions = await screen.findByRole("list", { name: "Suggested questions" })
  await userEvent.setup().click(within(suggestions).getByRole("button", { name: /Summarize the key points of Travel Policy\.pdf/ }))
  expect(api.chat).toHaveBeenCalledWith("Summarize the key points of Travel Policy.pdf")
})

it("reopens the conversation that was open before a reload", async () => {
  mockApi()
  sessionStorage.setItem("rag.conversations", JSON.stringify([
    { id: "c1", title: "Earlier question", messages: [{ id: "u1", role: "user", text: "Earlier question" }, { id: "a1", role: "assistant", question: "Earlier question", state: "done", response: answer() }] },
  ]))
  sessionStorage.setItem("rag.activeConversation", "c1")
  renderWorkspace(<ChatView />)
  expect(await screen.findByRole("heading", { level: 2, name: "Earlier question" })).toBeInTheDocument()
  expect(screen.getByRole("region", { name: "Sources" })).toBeInTheDocument()
})
