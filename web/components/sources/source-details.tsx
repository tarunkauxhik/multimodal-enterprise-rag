"use client"

import { EmojiText } from "@/components/emoji"
import { Passage } from "@/components/sources/passage"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Separator } from "@/components/ui/separator"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { useIsMobile } from "@/hooks/use-mobile"
import { CONTENT_TYPE, type SourceGroup } from "@/lib/citations"
import { cleanText, plain } from "@/lib/passage"

const typeLabel = (type: string) => CONTENT_TYPE[type] ?? "Text"

/** "Page 2 · Table": the page and the kinds of content cited from it. */
function meta(group: SourceGroup): string {
  const types = [...new Set(group.passages.map((p) => typeLabel(p.content_type)))]
  return [`Page ${group.page}`, ...types].join(" · ")
}

/** A passage's section heading, only when it adds something: not the document's own title and
 * not the same as the previous passage's. */
function sectionLabels(group: SourceGroup): (string | null)[] {
  const stem = group.document.replace(/\.pdf$/i, "").toLowerCase()
  let previous = ""
  return group.passages.map((p) => {
    const section = plain(cleanText(p.section_path.at(-1) ?? "")).trim()
    const useful = section !== "" && section.toLowerCase() !== stem && section !== previous
    previous = section
    return useful ? section : null
  })
}

/** Provenance for one cited document page. Every document-derived string is rendered as React
 * text (components/sources/passage.tsx), never as HTML or Markdown. */
function SourceBody({ group }: { group: SourceGroup }) {
  const labels = sectionLabels(group)
  if (group.passages.length === 0) return <p className="text-sm text-muted-foreground">The passage for this citation isn&apos;t available.</p>
  return (
    <div className="space-y-4">
      {group.passages.map((passage, i) => (
        <div key={i}>
          {i > 0 && <Separator className="mb-4" />}
          {labels[i] && (
            <p className="mb-1.5 text-xs font-medium text-muted-foreground">
              <EmojiText text={labels[i]!} />
            </p>
          )}
          <Passage text={passage.text} contentType={passage.content_type} />
        </div>
      ))}
    </div>
  )
}

function Header({ group }: { group: SourceGroup }) {
  return (
    <span className="flex min-w-0 items-start gap-2.5 text-left">
      <span className="mt-px grid size-5 shrink-0 place-items-center rounded border text-[11px] font-medium text-muted-foreground tabular-nums">
        {group.n}
      </span>
      <span className="min-w-0">
        <span className="block truncate text-sm font-medium text-foreground">
          <EmojiText text={group.document} />
        </span>
        <span className="block text-xs font-normal text-muted-foreground">{meta(group)}</span>
      </span>
    </span>
  )
}

/** Wraps a trigger (a citation marker or a source row): a popover on desktop, a bottom sheet on
 * touch-sized screens. */
export function SourceDetails({ group, children }: { group: SourceGroup; children: React.ReactNode }) {
  const mobile = useIsMobile()
  const label = `Source ${group.n}: ${group.document}, ${meta(group)}`
  if (mobile) {
    return (
      <Sheet>
        <SheetTrigger asChild>{children}</SheetTrigger>
        <SheetContent side="bottom" className="max-h-[85svh] gap-0">
          <SheetHeader className="border-b pr-12">
            <SheetTitle>
              <Header group={group} />
            </SheetTitle>
            <SheetDescription className="sr-only">{label}</SheetDescription>
          </SheetHeader>
          <div className="overflow-y-auto overscroll-contain p-4">
            <SourceBody group={group} />
          </div>
        </SheetContent>
      </Sheet>
    )
  }
  return (
    <Popover>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent align="start" className="w-136 max-w-[calc(100vw-2rem)] gap-0 p-0" aria-label={label}>
        <div className="border-b px-4 py-3">
          <Header group={group} />
        </div>
        <div className="max-h-104 overflow-y-auto overscroll-contain px-4 py-3">
          <SourceBody group={group} />
        </div>
      </PopoverContent>
    </Popover>
  )
}
