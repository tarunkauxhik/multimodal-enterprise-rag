"use client"

import { FileText, MoreHorizontal, RotateCw, Trash2, TriangleAlert, Upload } from "lucide-react"
import { useState } from "react"

import { StatusIcon } from "@/components/documents/status-icon"
import { Emoji, EmojiText } from "@/components/emoji"
import { PageHeader } from "@/components/page-header"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { useWorkspace } from "@/components/workspace-provider"
import { errorMessage, MAX_UPLOAD_MB } from "@/lib/api"
import { describe, isActive, PHASES, phaseOf, STATUS } from "@/lib/status"
import type { DocumentSummary } from "@/lib/types"
import { cn } from "@/lib/utils"

const HELP_TONE = { danger: "text-destructive", warning: "text-warning", neutral: "text-muted-foreground" } as const

export function DocumentsView() {
  const { documents, uploads, error, refresh, remove, openFilePicker } = useWorkspace()
  const [pendingDelete, setPendingDelete] = useState<DocumentSummary | null>(null)
  const [deleting, setDeleting] = useState(false)
  const ready = documents?.filter((d) => d.status === "ready").length ?? 0

  return (
    <>
      <PageHeader title="Documents">
        <Button size="sm" onClick={openFilePicker} className="pointer-coarse:h-10">
          <Upload />
          Add documents
        </Button>
      </PageHeader>

      <div className="mx-auto w-full max-w-4xl px-4 pt-6 pb-16 sm:px-6">
        <div className="mb-6">
          <h2 className="text-xl font-semibold tracking-tight">Documents</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            A shared workspace: everyone with access can see, use and remove these documents.
          </p>
        </div>

        <button
          type="button"
          onClick={openFilePicker}
          className="flex w-full items-center gap-3 rounded-lg border border-dashed px-4 py-4 text-left transition-colors hover:bg-muted/50 focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none"
        >
          <span className="grid size-10 shrink-0 place-items-center rounded-md bg-muted">
            <Emoji char="📥" decorative className="size-5" />
          </span>
          <span>
            <span className="block text-sm font-medium">Drop PDFs anywhere, or browse</span>
            <span className="block text-xs text-muted-foreground">PDF only, up to {MAX_UPLOAD_MB} MB each</span>
          </span>
        </button>

        {error && (
          <div role="alert" className="mt-6 flex items-center gap-3 rounded-lg border border-warning/30 bg-warning/5 px-4 py-3 text-sm">
            <TriangleAlert className="size-4 shrink-0 text-warning" aria-hidden />
            <span className="flex-1">{errorMessage(error, "load")}</span>
            <Button variant="outline" size="sm" onClick={() => void refresh()}>
              <RotateCw />
              Retry
            </Button>
          </div>
        )}

        <section aria-labelledby="documents-heading" className="mt-8">
          <div className="flex items-baseline justify-between border-b pb-2">
            <h3 id="documents-heading" className="text-sm font-medium">
              All documents
            </h3>
            {documents && documents.length > 0 && (
              <p className="text-xs text-muted-foreground tabular-nums">
                {ready} of {documents.length} ready
              </p>
            )}
          </div>

          <ul className="divide-y">
            {uploads.map((upload) => (
              <li key={upload.id} className="flex items-center gap-3 py-3">
                <FileText className="size-4 shrink-0 text-muted-foreground" aria-hidden />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">
                    <EmojiText text={upload.name} />
                  </p>
                  <p className="mt-0.5 text-xs text-muted-foreground tabular-nums">
                    {upload.state === "waiting" ? "Waiting to upload" : `Uploading · ${Math.round(upload.progress * 100)}%`}
                  </p>
                </div>
                {upload.state === "uploading" && (
                  <Progress value={upload.progress * 100} className="h-1 w-24" aria-label={`Uploading ${upload.name}`} />
                )}
              </li>
            ))}

            {documents === null &&
              !error &&
              [0, 1, 2].map((i) => (
                <li key={i} className="flex items-center gap-3 py-3.5" aria-hidden>
                  <Skeleton className="size-4" />
                  <div className="flex-1 space-y-1.5">
                    <Skeleton className="h-3.5 w-1/3" />
                    <Skeleton className="h-3 w-1/5" />
                  </div>
                </li>
              ))}

            {documents?.map((doc) => (
              <DocumentRow key={doc.document_id} doc={doc} onDelete={() => setPendingDelete(doc)} />
            ))}
          </ul>

          {documents?.length === 0 && uploads.length === 0 && (
            <div className="flex flex-col items-center py-12 text-center">
              <Emoji char="📂" decorative className="mb-3 size-8" />
              <p className="text-sm font-medium">No documents yet</p>
              <p className="mt-1 text-sm text-muted-foreground">Add a PDF to start asking questions about it.</p>
            </div>
          )}
        </section>
      </div>

      <AlertDialog open={pendingDelete !== null} onOpenChange={(open) => !open && !deleting && setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this document?</AlertDialogTitle>
            <AlertDialogDescription>
              <span className="font-medium text-foreground">
                <EmojiText text={pendingDelete?.source_name ?? ""} />
              </span> will be removed from the
              shared workspace and won&apos;t be used for future answers.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={deleting}
              onClick={async (event) => {
                event.preventDefault() // keep the dialog open until the request finishes
                if (!pendingDelete) return
                setDeleting(true)
                await remove(pendingDelete)
                setDeleting(false)
                setPendingDelete(null)
              }}
            >
              {deleting ? "Deleting…" : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}

function DocumentRow({ doc, onDelete }: { doc: DocumentSummary; onDelete: () => void }) {
  const status = STATUS[doc.status]
  const phase = phaseOf(doc)
  return (
    <li className="flex items-start gap-3 py-3">
      <FileText className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium" title={doc.source_name}>
          <EmojiText text={doc.source_name} />
        </p>
        <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1.5 text-foreground/80">
            <StatusIcon status={doc.status} className="size-3.5" />
            {status.label}
          </span>
          {(doc.status === "ready" || doc.status === "incomplete" || phase >= 0) && (
            <>
              <span aria-hidden>·</span>
              <span className="tabular-nums">{describe(doc)}</span>
            </>
          )}
        </div>
        {phase >= 0 && (
          <Progress
            value={((phase + 1) / PHASES.length) * 100}
            className="mt-2 h-1 max-w-56"
            aria-label={`Step ${phase + 1} of ${PHASES.length}: ${PHASES[phase]}`}
          />
        )}
        {status.help && <p className={cn("mt-1 text-xs", HELP_TONE[status.tone as keyof typeof HELP_TONE])}>{status.help}</p>}
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon-sm" aria-label={`Actions for ${doc.source_name}`} className="text-muted-foreground pointer-coarse:size-10">
            <MoreHorizontal />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem variant="destructive" disabled={isActive(doc)} onSelect={onDelete}>
            <Trash2 />
            {isActive(doc) ? "Delete (after processing)" : "Delete"}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </li>
  )
}
