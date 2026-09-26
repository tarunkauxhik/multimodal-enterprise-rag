import { cn } from "@/lib/utils"

const PETAL = "M12 10.4V1.8a8.6 8.6 0 0 1 8.6 8.6Z"

/** The mark: an aperture of four quarter-discs turning around an open centre, focusing on the one
 * passage that answers the question. Same drawing as app/icon.svg; one colour in both themes. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden className={cn("size-5 shrink-0", className)}>
      <g fill="#2f6bff">
        {[0, 90, 180, 270].map((angle) => (
          <path key={angle} d={PETAL} transform={`rotate(${angle} 12 12)`} />
        ))}
      </g>
    </svg>
  )
}
