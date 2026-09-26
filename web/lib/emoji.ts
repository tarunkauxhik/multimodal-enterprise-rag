// Every emoji in the UI is drawn with Apple's artwork (public/emoji, see scripts/copy-emoji.mjs),
// so it looks the same on every device.

// An emoji: a keycap, a flag, or an emoji-presentation (or FE0F-qualified) pictograph with an optional
// skin tone, joined by ZWJ into sequences. Plain text symbols such as © or ✓ without FE0F stay text.
const PICTO = String.raw`(?:\p{Emoji_Presentation}|\p{Extended_Pictographic}️)\p{Emoji_Modifier}?`
const EMOJI = new RegExp(
  String.raw`[#*0-9]️?⃣|[\u{1F1E6}-\u{1F1FF}]{2}|${PICTO}(?:‍(?:${PICTO}|\p{Extended_Pictographic}))*`,
  "gu",
)

/** File name (without .png) for an emoji: lowercase code points joined by "-", minus U+FE0F. */
export function emojiKey(emoji: string): string {
  return [...emoji].map((c) => c.codePointAt(0)!.toString(16)).filter((cp) => cp !== "fe0f").join("-")
}

export type Piece = { text: string } | { emoji: string }

/** Splits text into plain runs and emojis, in order. */
export function splitEmoji(text: string): Piece[] {
  const pieces: Piece[] = []
  let last = 0
  for (const match of text.matchAll(EMOJI)) {
    if (match.index > last) pieces.push({ text: text.slice(last, match.index) })
    pieces.push({ emoji: match[0] })
    last = match.index + match[0].length
  }
  if (last < text.length) pieces.push({ text: text.slice(last) })
  return pieces
}

// Minimal hast types: enough for the rehype transform below.
type HastText = { type: "text"; value: string }
type HastElement = { type: "element"; tagName: string; properties: Record<string, unknown>; children: HastNode[] }
type HastNode = HastText | HastElement | { type: string; children?: HastNode[] }

/** rehype plugin: replaces emojis in markdown text with <span data-emoji> placeholders that the
 * renderer turns into Apple emoji images. Code stays untouched. */
export function rehypeAppleEmoji() {
  const walk = (node: HastNode) => {
    if (!("children" in node) || !node.children) return
    if (node.type === "element" && ["code", "pre"].includes((node as HastElement).tagName)) return
    node.children = node.children.flatMap((child): HastNode[] => {
      if (child.type !== "text") {
        walk(child)
        return [child]
      }
      return splitEmoji((child as HastText).value).map((piece) =>
        "emoji" in piece
          ? { type: "element", tagName: "span", properties: { dataEmoji: piece.emoji }, children: [] }
          : { type: "text", value: piece.text },
      )
    })
  }
  return (tree: HastNode) => walk(tree)
}
