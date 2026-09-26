import { cn } from "@/lib/utils"

/** The mark: a cited point in square brackets, "[•]" (grounded answers with citations).
 * Inverts with the theme; the point keeps the brand blue. Same drawing as app/icon.svg. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden className={cn("size-6 shrink-0", className)}>
      <rect width="32" height="32" rx="8" className="fill-foreground" />
      <path d="M13 9.5H9.5v13H13M19 9.5h3.5v13H19" fill="none" strokeWidth="3" className="stroke-background" />
      <circle cx="16" cy="16" r="2.75" fill="#3d7bff" />
    </svg>
  )
}
