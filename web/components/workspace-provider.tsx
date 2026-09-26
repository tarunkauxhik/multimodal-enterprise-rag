"use client"

import { createContext, useCallback, useContext, useRef, useState } from "react"

import { Emoji } from "@/components/emoji"
import { useConversations } from "@/hooks/use-conversations"
import { useDocuments } from "@/hooks/use-documents"

type Workspace = ReturnType<typeof useDocuments> & {
  chat: ReturnType<typeof useConversations>
  openFilePicker: () => void
  commandOpen: boolean
  setCommandOpen: (open: boolean) => void
}

const WorkspaceContext = createContext<Workspace | null>(null)

export function useWorkspace(): Workspace {
  const workspace = useContext(WorkspaceContext)
  if (!workspace) throw new Error("useWorkspace must be used inside WorkspaceProvider")
  return workspace
}

/** Shared state for the sidebar, chat and documents views, plus one hidden file input and a
 * window-wide drop target for PDFs. */
export function WorkspaceProvider({ children }: { children: React.ReactNode }) {
  const documents = useDocuments()
  const chat = useConversations()
  const [commandOpen, setCommandOpen] = useState(false)
  const [dragging, setDragging] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const openFilePicker = useCallback(() => input.current?.click(), [])
  const { addFiles } = documents

  const hasFiles = (event: React.DragEvent) => event.dataTransfer.types.includes("Files")

  return (
    <WorkspaceContext.Provider value={{ ...documents, chat, openFilePicker, commandOpen, setCommandOpen }}>
      <div
        className="contents"
        onDragOver={(event) => {
          if (!hasFiles(event)) return
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={(event) => {
          if (event.relatedTarget === null) setDragging(false) // left the window
        }}
        onDrop={(event) => {
          if (!hasFiles(event)) return
          event.preventDefault()
          setDragging(false)
          addFiles(event.dataTransfer.files)
        }}
      >
        {children}
        {dragging && (
          <div className="pointer-events-none fixed inset-0 z-50 grid place-items-center bg-background/80 backdrop-blur-[2px] animate-in fade-in-0 duration-150">
            <div className="flex flex-col items-center rounded-lg border border-dashed border-foreground/25 px-10 py-8 text-center">
              <Emoji char="📥" decorative className="mb-3 size-8" />
              <p className="text-sm font-medium">Drop PDFs to add them</p>
              <p className="mt-1 text-sm text-muted-foreground">They&apos;ll be processed and added to the shared workspace.</p>
            </div>
          </div>
        )}
      </div>
      <input
        ref={input}
        type="file"
        accept="application/pdf,.pdf"
        multiple
        hidden
        aria-hidden
        tabIndex={-1}
        onChange={(event) => {
          if (event.target.files) addFiles(event.target.files)
          event.target.value = "" // allow picking the same file again
        }}
      />
    </WorkspaceContext.Provider>
  )
}
