import { EmojiText } from "@/components/emoji"
import { parsePassage } from "@/lib/passage"
import { numericColumns } from "@/lib/tables"

/** A source passage: prose as plain text, pipe tables as real tables. All content is React text. */
export function Passage({ text }: { text: string }) {
  return (
    <div className="space-y-3">
      {parsePassage(text).map((block, i) => {
        if (block.type === "text") {
          return (
            <p key={i} className="text-[13px] leading-relaxed whitespace-pre-wrap break-words text-foreground/90">
              <EmojiText text={block.text} />
            </p>
          )
        }
        const numeric = numericColumns(block.rows)
        const num = (j: number) => (numeric[j] ? "" : undefined)
        return (
          <div key={i} className="data-table overflow-x-auto rounded-md border">
            <table>
              {block.header && (
                <thead>
                  <tr>
                    {block.header.map((cell, j) => (
                      <th key={j} scope="col" data-numeric={num(j)}>
                        <EmojiText text={cell} />
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
                        <EmojiText text={cell} />
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
