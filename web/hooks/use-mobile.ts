import { useSyncExternalStore } from "react"

const QUERY = "(max-width: 1023px)" // matches the sidebar's lg: breakpoint

function subscribe(onChange: () => void) {
  const mql = window.matchMedia(QUERY)
  mql.addEventListener("change", onChange)
  return () => mql.removeEventListener("change", onChange)
}

export function useIsMobile(): boolean {
  return useSyncExternalStore(subscribe, () => window.matchMedia(QUERY).matches, () => false)
}
