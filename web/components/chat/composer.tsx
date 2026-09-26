"use client"

import { ArrowUp, LoaderCircle } from "lucide-react"
import { useEffect, useRef, useState } from "react"

import { Button } from "@/components/ui/button"
import { MAX_QUESTION_CHARS } from "@/lib/api"

export function Composer({ onSubmit, disabled, pending, placeholder, focusKey }: {
  onSubmit: (question: string) => void
  disabled: boolean
  pending: boolean
  placeholder: string
  focusKey?: string // focus again when this changes (e.g. a different conversation)
}) {
  const [value, setValue] = useState("")
  const textarea = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    // Desktop only: focusing on touch screens would pop the keyboard over the conversation.
    if (!disabled && window.matchMedia("(pointer: fine)").matches) textarea.current?.focus()
  }, [focusKey, disabled])
  const question = value.trim()
  const canSend = !disabled && !pending && question.length > 0 && question.length <= MAX_QUESTION_CHARS
  const nearLimit = value.length > MAX_QUESTION_CHARS * 0.9

  const submit = () => {
    if (!canSend) return
    onSubmit(question)
    setValue("")
  }

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        submit()
      }}
      className="rounded-xl border bg-background shadow-xs transition-[border-color,box-shadow] focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/20 has-[textarea:disabled]:bg-muted/40"
    >
      <label htmlFor="question" className="sr-only">
        Question
      </label>
      <textarea
        ref={textarea}
        id="question"
        rows={1}
        value={value}
        disabled={disabled}
        placeholder={placeholder}
        maxLength={MAX_QUESTION_CHARS}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault()
            submit()
          }
        }}
        className="field-sizing-content block max-h-48 min-h-12 w-full resize-none bg-transparent px-4 pt-3 pb-1 text-[15px] leading-6 outline-none pointer-coarse:text-base placeholder:text-muted-foreground disabled:cursor-not-allowed"
      />
      <div className="flex items-center justify-end gap-3 px-2 pb-2">
        {nearLimit && (
          <span className="text-xs text-muted-foreground tabular-nums" aria-live="polite">
            {value.length} / {MAX_QUESTION_CHARS}
          </span>
        )}
        <Button type="submit" size="icon" disabled={!canSend} aria-label={pending ? "Answering…" : "Send question"} className="rounded-lg pointer-coarse:size-10">
          {pending ? <LoaderCircle className="animate-spin" /> : <ArrowUp />}
        </Button>
      </div>
    </form>
  )
}
