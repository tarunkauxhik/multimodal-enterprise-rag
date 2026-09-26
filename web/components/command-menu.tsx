"use client"

import { FileText, MessageSquare, Monitor, Moon, PanelLeft, SquarePen, Sun, Trash2, Upload } from "lucide-react"
import { usePathname, useRouter } from "next/navigation"
import { useTheme } from "next-themes"
import { useEffect } from "react"

import { EmojiText } from "@/components/emoji"
import { useWorkspace } from "@/components/workspace-provider"
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
} from "@/components/ui/command"
import { useSidebar } from "@/components/ui/sidebar"
import { useModKey } from "@/hooks/use-mod-key"

export function CommandMenu() {
  const { commandOpen, setCommandOpen, chat, documents, openFilePicker } = useWorkspace()
  const { toggleSidebar } = useSidebar()
  const { setTheme } = useTheme()
  const router = useRouter()
  const pathname = usePathname()
  const mod = useModKey()

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() === "k" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault()
        setCommandOpen(!commandOpen)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [commandOpen, setCommandOpen])

  const run = (action: () => void) => () => {
    setCommandOpen(false)
    action()
  }
  const toChat = () => pathname !== "/" && router.push("/")

  return (
    <CommandDialog open={commandOpen} onOpenChange={setCommandOpen} title="Search and commands" description="Find a conversation or document, or run a command.">
      <CommandInput placeholder="Search conversations, documents and commands…" />
      <CommandList>
        <CommandEmpty>No results.</CommandEmpty>
        <CommandGroup heading="Actions">
          <CommandItem onSelect={run(() => { chat.newChat(); toChat() })}>
            <SquarePen />
            New chat
          </CommandItem>
          {chat.active && (
            <CommandItem onSelect={run(() => chat.remove(chat.active!.id))}>
              <Trash2 />
              Clear current conversation
            </CommandItem>
          )}
          <CommandItem onSelect={run(openFilePicker)}>
            <Upload />
            Add documents
          </CommandItem>
          <CommandItem onSelect={run(() => router.push("/documents"))}>
            <FileText />
            Manage documents
          </CommandItem>
          <CommandItem onSelect={run(toggleSidebar)}>
            <PanelLeft />
            Toggle sidebar
            <CommandShortcut>{mod} B</CommandShortcut>
          </CommandItem>
        </CommandGroup>
        <CommandGroup heading="Theme">
          <CommandItem onSelect={run(() => setTheme("light"))}><Sun />Light</CommandItem>
          <CommandItem onSelect={run(() => setTheme("dark"))}><Moon />Dark</CommandItem>
          <CommandItem onSelect={run(() => setTheme("system"))}><Monitor />System</CommandItem>
        </CommandGroup>
        {chat.conversations.length > 0 && (
          <CommandGroup heading="Conversations">
            {chat.conversations.map((c) => (
              <CommandItem key={c.id} value={`conversation ${c.id} ${c.title}`} onSelect={run(() => { chat.select(c.id); toChat() })}>
                <MessageSquare />
                <span className="truncate"><EmojiText text={c.title} /></span>
              </CommandItem>
            ))}
          </CommandGroup>
        )}
        {!!documents?.length && (
          <CommandGroup heading="Documents">
            {documents.map((doc) => (
              <CommandItem key={doc.document_id} value={`document ${doc.source_name}`} onSelect={run(() => router.push("/documents"))}>
                <FileText />
                <span className="truncate"><EmojiText text={doc.source_name} /></span>
              </CommandItem>
            ))}
          </CommandGroup>
        )}
      </CommandList>
    </CommandDialog>
  )
}
