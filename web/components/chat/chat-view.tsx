"use client"

import { RotateCw, Upload } from "lucide-react"
import Link from "next/link"
import { Fragment, useEffect, useRef, useState } from "react"

import { AssistantMessage } from "@/components/chat/assistant-message"
import { Composer } from "@/components/chat/composer"
import { Emoji, EmojiText } from "@/components/emoji"
import { PageHeader } from "@/components/page-header"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { useWorkspace } from "@/components/workspace-provider"
import { errorMessage } from "@/lib/api"
import { isActive } from "@/lib/status"

// Starter questions built from real document names: nothing is invented about their content.
const STARTERS = [
  { emoji: "📝", question: (name: string) => `Summarize the key points of ${name}` },
  { emoji: "📊", question: (name: string) => `What are the most important figures in ${name}?` },
  { emoji: "🗓️", question: (name: string) => `What dates, deadlines or time periods does ${name} mention?` },
]

export function ChatView() {
  const { documents, uploads, error, refresh, chat, openFilePicker } = useWorkspace()
  const readyDocs = documents?.filter((d) => d.status === "ready") ?? []
  const ready = readyDocs.length
  const processing = (documents ?? []).some(isActive) || uploads.length > 0
  const messages = chat.active?.messages ?? []

  // Scroll a newly asked question to the top of the view, like a new section of a document.
  const lastQuestion = useRef<HTMLHeadingElement>(null)
  useEffect(() => {
    lastQuestion.current?.scrollIntoView({ block: "start", behavior: "smooth" })
  }, [messages.length])

  // The header shows the conversation title only once the first question has scrolled away.
  const firstQuestion = useRef<HTMLHeadingElement>(null)
  const [titleVisible, setTitleVisible] = useState(false)
  useEffect(() => {
    const heading = firstQuestion.current
    if (!heading) return
    const observer = new IntersectionObserver(([entry]) => setTitleVisible(!entry.isIntersecting), { rootMargin: "-48px 0px 0px 0px" })
    observer.observe(heading)
    return () => observer.disconnect()
  }, [chat.active?.id])

  const lastUserIndex = messages.findLastIndex((m) => m.role === "user")
  const placeholder = ready > 0 ? "Ask about your documents…" : "Questions open once a document is ready"

  return (
    <>
      <PageHeader title={chat.active ? <EmojiText text={chat.active.title} /> : "New chat"} titleVisible={!chat.active || titleVisible} />
      <div className="flex flex-1 flex-col">
        {messages.length === 0 ? (
          <div className="flex flex-1 items-center justify-center px-4 pb-16">{emptyState()}</div>
        ) : (
          <div className="mx-auto w-full max-w-3xl flex-1 px-4 pt-6 pb-8 sm:px-6">
            {messages.map((message, i) => (
              <Fragment key={message.id}>
                {message.role === "user" ? (
                  <>
                    {i > 0 && <Separator className="my-8" />}
                    <h2
                      ref={(node) => {
                        if (i === 0) firstQuestion.current = node
                        if (i === lastUserIndex) lastQuestion.current = node
                      }}
                      className="mb-4 scroll-mt-16 text-lg font-semibold tracking-tight whitespace-pre-wrap break-words"
                    >
                      <EmojiText text={message.text} />
                    </h2>
                  </>
                ) : (
                  <AssistantMessage message={message} onRetry={() => chat.retry(message.id, message.question)} />
                )}
              </Fragment>
            ))}
            {/* room to scroll the latest question to the top while its answer is short */}
            <div aria-hidden className="h-[40svh]" />
          </div>
        )}

        <div className="sticky bottom-0 bg-background px-4 pt-2 pb-4 sm:px-6">
          <div className="mx-auto w-full max-w-3xl">
            <Composer onSubmit={chat.ask} disabled={ready === 0} pending={chat.pending} placeholder={placeholder} focusKey={chat.active?.id ?? "new"} />
            <p className="mt-2 hidden text-center text-xs text-muted-foreground sm:block">
              Answers cite the pages they come from. Check important facts in the source.
            </p>
          </div>
        </div>
      </div>
    </>
  )

  function emptyState() {
    if (documents === null && !error) return null // first load; the sidebar shows skeletons
    if (error) {
      return (
        <Centered emoji="🔌" title={errorMessage(error, "load")} text="Your documents and questions will be back as soon as it is.">
          <Button variant="outline" onClick={() => void refresh()}>
            <RotateCw />
            Try again
          </Button>
        </Centered>
      )
    }
    if (ready === 0 && processing) {
      return (
        <Centered emoji="⏳" title="Preparing your documents" text="You can ask questions as soon as a document is ready.">
          <Button variant="outline" asChild>
            <Link href="/documents">View progress</Link>
          </Button>
        </Centered>
      )
    }
    if (ready === 0) {
      return (
        <Centered emoji="📚" title="Ask questions about your documents" text="Upload a PDF to start a grounded conversation with your document set.">
          <Button onClick={openFilePicker}>
            <Upload />
            Add documents
          </Button>
        </Centered>
      )
    }
    const starters = readyDocs.slice(0, STARTERS.length).map((doc, i) => ({ ...STARTERS[i], text: STARTERS[i].question(doc.source_name) }))
    return (
      <Centered
        emoji="👋"
        title="What would you like to know?"
        text={`Answers come from ${ready} ready ${ready === 1 ? "document" : "documents"}, with page citations.`}
      >
        <ul aria-label="Suggested questions" className="flex w-full max-w-md flex-col gap-2 text-left">
          {starters.map((starter) => (
            <li key={starter.text}>
              <button
                type="button"
                onClick={() => chat.ask(starter.text)}
                className="flex w-full items-center gap-3 rounded-lg border px-3 py-2.5 text-left text-sm transition-colors pointer-coarse:py-3 hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none"
              >
                <Emoji char={starter.emoji} decorative />
                <span className="line-clamp-2 min-w-0">
                  <EmojiText text={starter.text} />
                </span>
              </button>
            </li>
          ))}
        </ul>
      </Centered>
    )
  }
}

function Centered({ emoji, title, text, children }: { emoji: string; title: string; text: string; children?: React.ReactNode }) {
  return (
    <div className="flex w-full max-w-md flex-col items-center text-center animate-in fade-in-0 duration-300">
      <Emoji char={emoji} decorative className="mb-4 size-10" />
      <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      <p className="mt-2 text-sm text-muted-foreground">{text}</p>
      {children && <div className="mt-6 flex w-full justify-center">{children}</div>}
    </div>
  )
}
