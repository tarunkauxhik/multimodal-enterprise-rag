import { cookies } from "next/headers"

import { AppSidebar } from "@/components/app-sidebar"
import { CommandMenu } from "@/components/command-menu"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"
import { WorkspaceProvider } from "@/components/workspace-provider"

export default async function WorkspaceLayout({ children }: { children: React.ReactNode }) {
  const defaultOpen = (await cookies()).get("sidebar_state")?.value !== "false"
  return (
    <WorkspaceProvider>
      <SidebarProvider defaultOpen={defaultOpen}>
        <AppSidebar />
        <SidebarInset className="min-w-0">{children}</SidebarInset>
        <CommandMenu />
      </SidebarProvider>
    </WorkspaceProvider>
  )
}
