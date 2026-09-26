import { EmojiText } from "@/components/emoji"
import { FIGURE_CAPTION, inline, parsePassage } from "@/lib/passage"
import { numericColumns } from "@/lib/tables"

/** Cleaned text with real super/subscripts. Every part is React text: nothing is parsed as HTML. */
function Rich({ text }: { text: string }) {
  return (
    <>
      {inline(text).map((part, i) =>
        part.style === "sup" ? (
          <sup key={i}>
            <EmojiText text={part.text} />
          </sup>
        ) : part.style === "sub" ? (
          <sub key={i}>
            <EmojiText text={part.text} />
          </sub>
        ) : (
          <EmojiText key={i} text={part.text} />
        ),
      )}
    </>
  )
}

/** A source passage, rendered by kind: prose as clean text with headings, pipe tables as real
 * tables, and a figure's leading caption as its title. */
export function Passage({ text, contentType = "text" }: { text: string; contentType?: string }) {
  const blocks = parsePassage(text)
  return (
    <div className="space-y-2.5">
      {blocks.map((block, i) => {
        if (block.type === "heading") {
          return (
            <p key={i} className="text-[13px] font-medium text-foreground">
              <Rich text={block.text} />
            </p>
          )
        }
        if (block.type === "text") {
          const [first, ...rest] = block.text.split("\n")
          if (contentType === "figure" && i === 0 && FIGURE_CAPTION.test(first)) {
            return (
              <div key={i}>
                <p className="text-[13px] font-medium text-foreground">
                  <Rich text={first} />
                </p>
                {rest.length > 0 && (
                  <p className="mt-1 text-[13px] leading-relaxed whitespace-pre-wrap wrap-break-word text-foreground/85">
                    <Rich text={rest.join("\n")} />
                  </p>
                )}
              </div>
            )
          }
          return (
            <p key={i} className="text-[13px] leading-relaxed whitespace-pre-wrap wrap-break-word text-foreground/85">
              <Rich text={block.text} />
            </p>
          )
        }
        const numeric = numericColumns(block.rows)
        const num = (j: number) => (numeric[j] ? "" : undefined)
        return (
          <div key={i} className="data-table max-w-full overflow-x-auto rounded-md border">
            <table>
              {block.header && (
                <thead>
                  <tr>
                    {block.header.map((cell, j) => (
                      <th key={j} scope="col" data-numeric={num(j)}>
                        <Rich text={cell} />
                      </th>
                    ))}
                  </tr>
                </thead>
              )}
              <tbody>
                {block.rows.map((row, r) => (
                  <tr key={r}>
                    {row.map((cell, j) => (
                      <td key={j} data-numeric={num(j)}>
                        <Rich text={cell} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      })}
    </div>
  )
}
