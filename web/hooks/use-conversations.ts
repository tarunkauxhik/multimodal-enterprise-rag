"use client"

import { useCallback, useEffect, useState } from "react"

import { api, errorMessage, HISTORY_ANSWER_CHARS, HISTORY_TURNS, MAX_QUESTION_CHARS } from "@/lib/api"
import { replyKind, type ChatResponse, type ChatTurn } from "@/lib/types"

export type Message =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "assistant"; question: string; state: "pending" }
  | { id: string; role: "assistant"; question: string; state: "done"; response: ChatResponse }
  | { id: string; role: "assistant"; question: string; state: "error"; error: string }

export interface Conversation {
  id: string
  title: string
  messages: Message[]
}

const STORAGE_KEY = "rag.conversations" // this browser tab only; the backend keeps no history
const ACTIVE_KEY = "rag.activeConversation" // reopen the same conversation after a reload

/** A conversation that so far is only small talk ("hi") takes its title from the first real question. */
function onlySmallTalk(conversation: Conversation): boolean {
  return conversation.messages.every((m) => m.role === "user" || (m.state === "done" && replyKind(m.response) === "conversation"))
}

function load(): Conversation[] {
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) ?? "[]") as Conversation[]
    // A reply still pending when the tab was reloaded will never arrive.
    return saved.map((c) => ({
      ...c,
      messages: c.messages.map((m) =>
        m.role === "assistant" && m.state === "pending" ? { ...m, state: "error", error: "This answer was interrupted." } : m,
      ),
    }))
  } catch {
    return []
  }
}

/** The last few answered exchanges before a question, sent with it so the API can resolve a follow-up
 * ("tell me more", "what about page 4?"). Small talk and failed requests carry no context. */
export function chatHistory(messages: Message[]): ChatTurn[] {
  return messages
    .filter((m): m is Extract<Message, { state: "done" }> =>
      m.role === "assistant" && m.state === "done" && replyKind(m.response) !== "conversation" && m.question.trim() !== "",
    )
    .slice(-HISTORY_TURNS)
    .map((m) => ({ question: m.question.slice(0, MAX_QUESTION_CHARS), answer: m.response.answer.slice(0, HISTORY_ANSWER_CHARS) }))
}

/** History lives in this tab (sessionStorage); each question is sent with its last few exchanges (chatHistory). */
export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    const saved = load()
    let active: string | null = null
    try {
      active = sessionStorage.getItem(ACTIVE_KEY)
    } catch {}
    /* eslint-disable react-hooks/set-state-in-effect -- sessionStorage is only readable after mount */
    setConversations(saved)
    if (active && saved.some((c) => c.id === active)) setActiveId(active)
    setLoaded(true)
    /* eslint-enable react-hooks/set-state-in-effect */
  }, [])
  useEffect(() => {
    if (!loaded) return // never overwrite saved history before it has been read
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(conversations))
      if (activeId) sessionStorage.setItem(ACTIVE_KEY, activeId)
      else sessionStorage.removeItem(ACTIVE_KEY)
    } catch {}
  }, [conversations, activeId, loaded])

  const active = conversations.find((c) => c.id === activeId) ?? null

  const patch = (conversationId: string, messageId: string, message: Message) =>
    setConversations((list) =>
      list.map((c) =>
        c.id === conversationId ? { ...c, messages: c.messages.map((m) => (m.id === messageId ? message : m)) } : c,
      ),
    )

  const answer = useCallback(async (conversationId: string, messageId: string, question: string, history: ChatTurn[]) => {
    try {
      const response = await api.chat(question, history)
      patch(conversationId, messageId, { id: messageId, role: "assistant", question, state: "done", response })
    } catch (err) {
      patch(conversationId, messageId, { id: messageId, role: "assistant", question, state: "error", error: errorMessage(err, "chat") })
    }
  }, [])

  const ask = useCallback(
    (question: string) => {
      const conversationId = activeId ?? crypto.randomUUID()
      const messageId = crypto.randomUUID()
      const history = chatHistory(active?.messages ?? []) // a new conversation has none
      const turn: Message[] = [
        { id: crypto.randomUUID(), role: "user", text: question },
        { id: messageId, role: "assistant", question, state: "pending" },
      ]
      setConversations((list) =>
        list.some((c) => c.id === conversationId)
          ? list.map((c) =>
              c.id === conversationId
                ? { ...c, title: onlySmallTalk(c) ? question.slice(0, 80) : c.title, messages: [...c.messages, ...turn] }
                : c,
            )
          : [{ id: conversationId, title: question.slice(0, 80), messages: turn }, ...list],
      )
      setActiveId(conversationId)
      void answer(conversationId, messageId, question, history)
    },
    [activeId, active, answer],
  )

  const retry = useCallback(
    (messageId: string, question: string) => {
      if (!activeId) return
      const messages = active?.messages ?? []
      const history = chatHistory(messages.slice(0, messages.findIndex((m) => m.id === messageId))) // turns before this one
      patch(activeId, messageId, { id: messageId, role: "assistant", question, state: "pending" })
      void answer(activeId, messageId, question, history)
    },
    [activeId, active, answer],
  )

  const remove = useCallback((id: string) => {
    setConversations((list) => list.filter((c) => c.id !== id))
    setActiveId((current) => (current === id ? null : current))
  }, [])

  return {
    conversations,
    active,
    pending: active?.messages.some((m) => m.role === "assistant" && m.state === "pending") ?? false,
    ask,
    retry,
    select: setActiveId,
    newChat: useCallback(() => setActiveId(null), []),
    remove,
  }
}
