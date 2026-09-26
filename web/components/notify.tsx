import { toast } from "sonner"

import { Emoji } from "@/components/emoji"

/** A toast whose icon is an Apple emoji. Errors keep sonner's error semantics. */
export function notify(emoji: string, message: string, kind: "default" | "error" = "default") {
  const show = kind === "error" ? toast.error : toast
  show(message, { icon: <Emoji char={emoji} decorative /> })
}
