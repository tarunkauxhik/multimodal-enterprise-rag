import { useSyncExternalStore } from "react"

const subscribe = () => () => {}

/** "⌘" on Apple platforms, "Ctrl" elsewhere (and during server rendering). */
export function useModKey(): string {
  return useSyncExternalStore(
    subscribe,
    () => (/Mac|iPhone|iPad/.test(navigator.platform) ? "⌘" : "Ctrl"),
    () => "Ctrl",
  )
}
