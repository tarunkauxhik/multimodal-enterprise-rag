"use client"

import { Check, Copy, RotateCw } from "lucide-react"
import { useEffect, useMemo, useState } from "react"

import { AnswerMarkdown } from "@/components/chat/answer-markdown"
import { Emoji, EmojiText } from "@/components/emoji"
import { SourceDetails } from "@/components/sources/source-details"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { Message } from "@/hooks/use-conversations"
import { numberCitations, type SourceGroup } from "@/lib/citations"
import { passagePreview } from "@/lib/passage"
import type { ChatResponse } from "@/lib/types"

type AssistantMessage = Extract<Message, { role: "assistant" }>

export function AssistantMessage({ message, onRetry }: { message: AssistantMessage; onRetry: () => void }) {
  if (message.state === "pending") return <Pending />
  if (message.state === "error") {
    return (
      <div role="alert" className="flex flex-wrap items-center gap-x-3 gap-y-2 text-sm">
        <span className="inline-flex items-center gap-2 text-destructive">
          <Emoji char="⚠️" decorative />
          {message.error}
        </span>
        <Button variant="outline" size="sm" onClick={onRetry} className="pointer-coarse:h-10">
          <RotateCw />
          Retry
        </Button>
      </div>
    )
  }
  if (message.response.abstained) {
    return (
      <div className="flex gap-3 text-[15px] leading-7">
        <Emoji char="🤔" decorative className="mt-1" />
        <div>
          <p className="text-foreground/80">
            <EmojiText text={message.response.answer} />
          </p>
          <p className="text-sm text-muted-foreground">Try rephrasing, or add a document that covers this topic.</p>
        </div>
      </div>
    )
  }
  return <Answer response={message.response} />
}

/** Honest progress: the API answers in one request, so only elapsed time is shown, not fake stages. */
function Pending() {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    const timer = window.setInterval(() => setSeconds((s) => s + 1), 1000)
    return () => window.clearInterval(timer)
  }, [])
  return (
    <div aria-live="polite" aria-busy="true" className="space-y-2.5 py-1">
      <p className="flex items-center gap-2 text-sm text-muted-foreground">
        <Emoji char="🔎" decorative />
        Searching your documents…
        {seconds >= 3 && <span className="tabular-nums">{seconds}s</span>}
      </p>
      <Skeleton className="h-3.5 w-11/12" />
      <Skeleton className="h-3.5 w-4/5" />
      <Skeleton className="h-3.5 w-2/3" />
    </div>
  )
}

function Answer({ response }: { response: ChatResponse }) {
  const { markdown, groups } = useMemo(() => numberCitations(response), [response])
  const byNumber = new Map(groups.map((g) => [g.n, g]))

  return (
    <div>
      <AnswerMarkdown
        markdown={markdown}
        renderCitation={(n) => {
          const group = byNumber.get(n)
          return group ? <CitationMarker group={group} /> : null
        }}
      />
      {groups.length > 0 && (
        <section aria-label="Sources" className="mt-6">
          <h3 className="mb-1.5 text-xs font-medium text-muted-foreground">Sources</h3>
          <ol className="flex flex-col">
            {groups.map((group) => (
              <li key={group.n}>
                <SourceDetails group={group}>
                  <button
                    type="button"
                    className="-mx-2 flex w-[calc(100%+1rem)] items-start gap-2.5 rounded-md px-2 py-2 text-left text-sm transition-colors pointer-coarse:py-2.5 hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none"
                  >
                    <span className="mt-px grid size-5 shrink-0 place-items-center rounded border text-[11px] font-medium text-muted-foreground tabular-nums">
                      {group.n}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="flex min-w-0 items-baseline gap-2">
                        <span className="truncate font-medium">
                          <EmojiText text={group.document} />
                        </span>
                        <span className="shrink-0 text-muted-foreground tabular-nums">p. {group.page}</span>
                        {group.passages.length > 1 && (
                          <span className="shrink-0 text-xs text-muted-foreground">· {group.passages.length} passages</span>
                        )}
                      </span>
                      {group.passages[0] && (
                        <span className="mt-0.5 block truncate text-xs text-muted-foreground">
                          <EmojiText text={passagePreview(group.passages[0].text)} />
                        </span>
                      )}
                    </span>
                  </button>
                </SourceDetails>
              </li>
            ))}
          </ol>
        </section>
      )}
      <CopyButton text={response.answer} />
    </div>
  )
}

function CitationMarker({ group }: { group: SourceGroup }) {
  return (
    <SourceDetails group={group}>
      <button
        type="button"
        aria-label={`Source ${group.n}: ${group.document}, page ${group.page}`}
        className="relative -top-px mx-0.5 inline-flex h-4.5 min-w-4.5 items-center justify-center rounded border border-brand/25 bg-brand/8 px-1 align-baseline text-[11px] leading-none font-medium text-brand tabular-nums transition-colors after:absolute after:-inset-x-1 after:-inset-y-3 after:content-[''] hover:bg-brand/15 focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none"
      >
        {group.n}
      </button>
    </SourceDetails>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="mt-3 flex">
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="icon-sm"
            className="-ml-1.5 text-muted-foreground pointer-coarse:size-10"
            aria-label={copied ? "Copied" : "Copy answer"}
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(text)
                setCopied(true)
                window.setTimeout(() => setCopied(false), 1500)
              } catch {}
            }}
          >
            {copied ? <Check /> : <Copy />}
          </Button>
        </TooltipTrigger>
        <TooltipContent>{copied ? "Copied" : "Copy answer"}</TooltipContent>
      </Tooltip>
    </div>
  )
}
