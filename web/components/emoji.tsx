"use client"

import { memo, useState } from "react"

import { emojiKey, splitEmoji } from "@/lib/emoji"
import { cn } from "@/lib/utils"

/** One emoji drawn with Apple's artwork. `decorative` hides it from screen readers; otherwise its
 * alt text is the emoji itself. Falls back to the text character if the image is missing. */
export function Emoji({ char, decorative = false, className }: { char: string; decorative?: boolean; className?: string }) {
  const [failed, setFailed] = useState(false)
  if (failed) return <span aria-hidden={decorative || undefined}>{char}</span>
  return (
    // eslint-disable-next-line @next/next/no-img-element -- 3,700 tiny static PNGs; next/image adds nothing here
    <img
      src={`/emoji/${emojiKey(char)}.png`}
      alt={decorative ? "" : char}
      aria-hidden={decorative || undefined}
      draggable={false}
      loading="lazy"
      decoding="async"
      onError={() => setFailed(true)}
      className={cn("inline-block size-[1.2em] shrink-0 align-[-0.22em] select-none", className)}
    />
  )
}

/** Plain text (document-derived or user-typed) with any emojis in it drawn as Apple emojis.
 * Everything else stays React text: nothing is parsed as HTML or Markdown. */
export const EmojiText = memo(function EmojiText({ text }: { text: string }) {
  const pieces = splitEmoji(text)
  if (pieces.length === 1 && "text" in pieces[0]) return <>{text}</>
  return <>{pieces.map((piece, i) => ("emoji" in piece ? <Emoji key={i} char={piece.emoji} /> : piece.text))}</>
})
