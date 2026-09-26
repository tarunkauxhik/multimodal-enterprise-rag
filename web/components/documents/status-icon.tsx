import { CircleAlert, CircleCheck, CircleDashed, CircleMinus, Clock, LoaderCircle, TriangleAlert } from "lucide-react"

import { cn } from "@/lib/utils"
import { STATUS } from "@/lib/status"
import type { DocumentStatus } from "@/lib/types"

const ICONS = {
  ready: [CircleCheck, "text-success"],
  queued: [Clock, "text-muted-foreground"],
  processing: [LoaderCircle, "text-brand animate-spin"],
  failed: [CircleAlert, "text-destructive"],
  incomplete: [TriangleAlert, "text-warning"],
  empty: [CircleMinus, "text-muted-foreground"],
  uploading: [CircleDashed, "text-brand"],
} as const

/** Status shown by shape and color; pair it with a text label (or pass `label`) so color is never the only cue. */
export function StatusIcon({ status, label, className }: { status: DocumentStatus | "uploading"; label?: boolean; className?: string }) {
  const [Icon, tone] = ICONS[status]
  const text = status === "uploading" ? "Uploading" : STATUS[status].label
  return <Icon className={cn("size-4 shrink-0", tone, className)} aria-label={label ? text : undefined} aria-hidden={!label} role={label ? "img" : undefined} />
}
