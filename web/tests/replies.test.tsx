import { act, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import axe from "axe-core"
import { expect, it, vi } from "vitest"

import { ChatView } from "@/components/chat/chat-view"
import type { ChatResponse } from "@/lib/types"
import { answer, mockApi, renderWorkspace } from "./helpers"

const reply = (kind: ChatResponse["kind"], text: string, suggestions: string[] = []): ChatResponse => ({
  answer: text, abstained: kind === "abstain" || kind === "out_of_scope", citations: [], sources: [], kind, suggestions,
})

async function ask(question: string) {
  const user = userEvent.setup()
  const box = await screen.findByRole("textbox", { name: "Question" })
  await waitFor(() => expect(box).toBeEnabled())
  await user.type(box, question)
  await user.keyboard("{Enter}")
  return user
}

it("answers small talk as a light exchange, without sources or copy", async () => {
  const api = mockApi()
  api.chat.mockResolvedValue(reply("conversation", "Hey 👋 What are we digging into?"))
  renderWorkspace(<ChatView />)
  await ask("hi")
  expect(await screen.findByText(/What are we digging into\?/)).toBeInTheDocument()
  expect(screen.getByRole("img", { name: "👋" })).toBeInTheDocument()
  expect(screen.queryByRole("region", { name: "Sources" })).not.toBeInTheDocument()
  expect(screen.queryByRole("button", { name: "Copy answer" })).not.toBeInTheDocument()
  expect(screen.getByRole("heading", { level: 2, name: "hi" })).toHaveClass("text-[15px]") // not a document-style heading
})

it("offers one-tap follow-ups with a clarification", async () => {
  const api = mockApi()
  api.chat.mockResolvedValueOnce(reply("clarify", "Do you mean CI (Compound Interest)? I can pull up the relevant section.", ["What is Compound Interest?"]))
  renderWorkspace(<ChatView />)
  const user = await ask("ci")
  const followUps = await screen.findByRole("group", { name: "Suggested follow-ups" })
  await user.click(within(followUps).getByRole("button", { name: "What is Compound Interest?" }))
  expect(api.chat).toHaveBeenLastCalledWith("What is Compound Interest?", [
    { question: "ci", answer: "Do you mean CI (Compound Interest)? I can pull up the relevant section." }, // the clarification is context
  ])
})

it("renders workspace answers as formatted text without Markdown symbols", async () => {
  const api = mockApi()
  api.chat.mockResolvedValue(reply("workspace", "You have 2 documents:\n\n- **Annual Report.pdf**, 12 indexed pages\n- **Policy.pdf**, 3 indexed pages"))
  renderWorkspace(<ChatView />)
  await ask("what documents are uploaded?")
  const name = await screen.findByText("Annual Report.pdf", { selector: "strong" })
  expect(name.closest("li")).toHaveTextContent("Annual Report.pdf, 12 indexed pages")
  expect(screen.getByRole("main")).not.toHaveTextContent("**")
})

it("explains out-of-scope questions differently from missing evidence", async () => {
  const api = mockApi()
  api.chat.mockResolvedValue(reply("out_of_scope", "That’s outside the uploaded docs, so I’d rather not guess."))
  renderWorkspace(<ChatView />)
  await ask("What is the capital of France?")
  expect(await screen.findByText(/outside the uploaded docs/)).toBeInTheDocument()
  expect(screen.queryByText(/Try rephrasing/)).not.toBeInTheDocument()
})

it("opens a citation from the keyboard and returns focus when it closes", async () => {
  mockApi()
  renderWorkspace(<ChatView />)
  const user = await ask("What was revenue?")
  const [marker] = await screen.findAllByRole("button", { name: /^Source 1: Annual Report\.pdf/ })
  marker.focus()
  await user.keyboard("{Enter}")
  const details = await screen.findByRole("dialog", { name: /Source 1: Annual Report\.pdf, Page 4 · Table · Text/ })
  expect(within(details).getByRole("table")).toBeInTheDocument()
  const results = await axe.run(document.body, { rules: { "color-contrast": { enabled: false } } })
  expect(results.violations.map((v) => v.id)).toEqual([])
  await user.keyboard("{Escape}")
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
  expect(marker).toHaveFocus()
})

it("shows a quiet indicator first, and the search message only for slower answers", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  const api = mockApi()
  api.chat.mockReturnValue(new Promise<ChatResponse>(() => {})) // never answers
  renderWorkspace(<ChatView />)
  await ask("What was revenue?")
  expect(await screen.findByLabelText("Thinking")).toBeInTheDocument()
  expect(screen.queryByText(/Searching your documents/)).not.toBeInTheDocument()
  await act(() => vi.advanceTimersByTimeAsync(800))
  expect(screen.getByText(/Searching your documents/)).toBeInTheDocument()
  vi.useRealTimers()
})

it("still renders answers saved before reply kinds existed", async () => {
  mockApi()
  const legacy = answer()
  sessionStorage.setItem("rag.conversations", JSON.stringify([
    { id: "c1", title: "Old", messages: [{ id: "u1", role: "user", text: "Old" }, { id: "a1", role: "assistant", question: "Old", state: "done", response: legacy }] },
  ]))
  sessionStorage.setItem("rag.activeConversation", "c1")
  renderWorkspace(<ChatView />)
  expect(await screen.findByRole("region", { name: "Sources" })).toBeInTheDocument()
})

it("titles a conversation that began with small talk after its first real question", async () => {
  const api = mockApi()
  api.chat.mockResolvedValueOnce(reply("conversation", "Hey 👋 What are we digging into?"))
  renderWorkspace(<ChatView />)
  const user = await ask("hi")
  expect(await screen.findByRole("button", { name: "hi" })).toBeInTheDocument() // sidebar conversation
  await user.type(screen.getByRole("textbox", { name: "Question" }), "What was revenue?{Enter}")
  expect(await screen.findByRole("button", { name: "What was revenue?" })).toBeInTheDocument()
  expect(screen.queryByRole("button", { name: "hi" })).not.toBeInTheDocument()
})
