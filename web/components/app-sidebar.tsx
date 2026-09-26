"use client"

import { FileText, Monitor, Moon, MoreHorizontal, Plus, Search, SquarePen, Sun, Trash2, TriangleAlert } from "lucide-react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { useTheme } from "next-themes"

import { StatusIcon } from "@/components/documents/status-icon"
import { EmojiText } from "@/components/emoji"
import { LogoMark } from "@/components/logo"
import { useWorkspace } from "@/components/workspace-provider"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Kbd } from "@/components/ui/kbd"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupAction,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  useSidebar,
} from "@/components/ui/sidebar"
import { useModKey } from "@/hooks/use-mod-key"
import { errorMessage } from "@/lib/api"
import { describe } from "@/lib/status"

export function AppSidebar() {
  const { documents, uploads, error, chat, openFilePicker, setCommandOpen } = useWorkspace()
  const { setOpenMobile } = useSidebar()
  const pathname = usePathname()
  const router = useRouter()
  const mod = useModKey()

  const goToChat = (conversationId: string | null) => {
    if (conversationId) chat.select(conversationId)
    else chat.newChat()
    setOpenMobile(false)
    if (pathname !== "/") router.push("/")
  }

  return (
    <Sidebar collapsible="icon" role="navigation" aria-label="Workspace">
      <SidebarHeader>
        <div className="flex h-8 items-center gap-2 px-2 group-data-[collapsible=icon]:px-0">
          <LogoMark className="group-data-[collapsible=icon]:mx-auto" />
          <span className="truncate text-sm font-semibold tracking-tight group-data-[collapsible=icon]:hidden">Enterprise RAG</span>
        </div>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton tooltip="New chat" onClick={() => goToChat(null)} className="pointer-coarse:h-10">
              <SquarePen />
              <span>New chat</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton tooltip="Search and commands" onClick={() => setCommandOpen(true)} className="pointer-coarse:h-10">
              <Search />
              <span>Search</span>
              <Kbd className="ml-auto group-data-[collapsible=icon]:hidden">{mod} K</Kbd>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton asChild tooltip="Documents" isActive={pathname === "/documents"} className="pointer-coarse:h-10">
              <Link href="/documents" onClick={() => setOpenMobile(false)}>
                <FileText />
                <span>Documents</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
          <SidebarGroupLabel>Conversations</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {chat.conversations.length === 0 && <p className="px-2 py-1 text-xs text-muted-foreground">No conversations yet</p>}
              {chat.conversations.map((conversation) => (
                <SidebarMenuItem key={conversation.id}>
                  <SidebarMenuButton
                    isActive={pathname === "/" && chat.active?.id === conversation.id}
                    onClick={() => goToChat(conversation.id)}
                    className="pointer-coarse:h-10"
                  >
                    <span>
                      <EmojiText text={conversation.title} />
                    </span>
                  </SidebarMenuButton>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <SidebarMenuAction showOnHover aria-label={`Actions for ${conversation.title}`}>
                        <MoreHorizontal />
                      </SidebarMenuAction>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent side="right" align="start">
                      <DropdownMenuItem onSelect={() => chat.remove(conversation.id)}>
                        <Trash2 />
                        Remove conversation
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
          <SidebarGroupLabel>Documents</SidebarGroupLabel>
          <SidebarGroupAction title="Add documents" aria-label="Add documents" onClick={openFilePicker}>
            <Plus />
          </SidebarGroupAction>
          <SidebarGroupContent>
            <SidebarMenu>
              {documents === null && !error && ["70%", "55%", "80%"].map((width) => (
                  <div key={width} className="flex h-8 items-center gap-2 px-2" aria-hidden>
                    <Skeleton className="size-4 rounded-md" />
                    <Skeleton className="h-3.5" style={{ width }} />
                  </div>
                ))}
              {documents?.length === 0 && uploads.length === 0 && (
                <p className="px-2 py-1 text-xs text-muted-foreground">No documents yet</p>
              )}
              {uploads.map((upload) => (
                <SidebarMenuItem key={upload.id}>
                  <div className="flex items-start gap-2 px-2 py-1.5 text-sm">
                    <StatusIcon status="uploading" className="mt-0.5" />
                    <div className="min-w-0">
                      <p className="truncate">
                        <EmojiText text={upload.name} />
                      </p>
                      <p className="text-xs text-muted-foreground tabular-nums">
                        {upload.state === "waiting" ? "Waiting to upload" : `Uploading ${Math.round(upload.progress * 100)}%`}
                      </p>
                    </div>
                  </div>
                </SidebarMenuItem>
              ))}
              {documents?.map((doc) => (
                <SidebarMenuItem key={doc.document_id}>
                  <SidebarMenuButton asChild size="lg" className="h-auto py-1.5" title={doc.source_name}>
                    <Link href="/documents" onClick={() => setOpenMobile(false)}>
                      <StatusIcon status={doc.status} className="mt-0.5 self-start" />
                      <span className="min-w-0">
                        <span className="block truncate">
                          <EmojiText text={doc.source_name} />
                        </span>
                        <span className="block truncate text-xs text-muted-foreground tabular-nums">{describe(doc)}</span>
                      </span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          {error && (
            <SidebarMenuItem>
              <SidebarMenuButton tooltip={errorMessage(error, "load")} className="text-warning hover:text-warning" asChild>
                <Link href="/documents">
                  <TriangleAlert />
                  <span>{errorMessage(error, "load")}</span>
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
          )}
          <SidebarMenuItem>
            <ThemeMenu />
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}

function ThemeMenu() {
  const { theme, setTheme } = useTheme()
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <SidebarMenuButton tooltip="Theme" className="pointer-coarse:h-10">
          <Sun className="dark:hidden" />
          <Moon className="hidden dark:block" />
          <span>Theme</span>
        </SidebarMenuButton>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="top" align="start" className="w-44">
        <DropdownMenuLabel>Theme</DropdownMenuLabel>
        <DropdownMenuRadioGroup value={theme} onValueChange={setTheme}>
          <DropdownMenuRadioItem value="light"><Sun />Light</DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="dark"><Moon />Dark</DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="system"><Monitor />System</DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
