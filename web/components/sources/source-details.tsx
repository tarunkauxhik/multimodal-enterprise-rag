"use client"

import { FileText } from "lucide-react"

import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Separator } from "@/components/ui/separator"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { useIsMobile } from "@/hooks/use-mobile"
import { Emoji, EmojiText } from "@/components/emoji"
import { Passage } from "@/components/sources/passage"
import { CONTENT_EMOJI, CONTENT_TYPE, type SourceGroup } from "@/lib/citations"

/** Provenance for one cited document page. Every document-derived string is rendered as React
 * text, never as HTML or Markdown. */
function SourceBody({ group }: { group: SourceGroup }) {
  return (
    <div className="space-y-4">
      {group.passages.length === 0 && <p className="text-sm text-muted-foreground">The passage for this citation isn&apos;t available.</p>}
      {group.passages.map((passage, i) => (
        <div key={i}>
          {i > 0 && <Separator className="mb-4" />}
          <p className="mb-2 flex items-center gap-1.5 text-xs text-muted-foreground">
            <Emoji char={CONTENT_EMOJI[passage.content_type] ?? CONTENT_EMOJI.text} decorative />
            <span>
              {CONTENT_TYPE[passage.content_type] ?? "Text"}
              {passage.section_path.length > 0 && (
                <>
                  {" · "}
                  <EmojiText text={passage.section_path.join(" › ")} />
                </>
              )}
            </span>
          </p>
          <Passage text={passage.text} />
        </div>
      ))}
    </div>
  )
}

function Title({ group }: { group: SourceGroup }) {
  return (
    <span className="flex min-w-0 items-center gap-2">
      <FileText className="size-4 shrink-0 text-muted-foreground" aria-hidden />
      <span className="truncate">
        <EmojiText text={group.document} />
      </span>
      <span className="shrink-0 font-normal text-muted-foreground">Page {group.page}</span>
    </span>
  )
}

/** Wraps a trigger (a citation marker or a source row): a popover on desktop, a bottom sheet on mobile. */
export function SourceDetails({ group, children }: { group: SourceGroup; children: React.ReactNode }) {
  const mobile = useIsMobile()
  if (mobile) {
    return (
      <Sheet>
        <SheetTrigger asChild>{children}</SheetTrigger>
        <SheetContent side="bottom" className="max-h-[80svh] gap-0">
          <SheetHeader className="border-b">
            <SheetTitle className="text-sm">
              <Title group={group} />
            </SheetTitle>
            <SheetDescription className="sr-only">Source {group.n}</SheetDescription>
          </SheetHeader>
          <div className="overflow-y-auto p-4">
            <SourceBody group={group} />
          </div>
        </SheetContent>
      </Sheet>
    )
  }
  return (
    <Popover>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent align="start" className="w-[32rem] max-w-[calc(100vw-2rem)] gap-0 p-0" aria-label={`Source ${group.n}: ${group.document}, page ${group.page}`}>
        <div className="border-b px-4 py-2.5 text-sm font-medium">
          <Title group={group} />
        </div>
        <div className="max-h-96 overflow-y-auto px-4 py-3">
          <SourceBody group={group} />
        </div>
      </PopoverContent>
    </Popover>
  )
}
