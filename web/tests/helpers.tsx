import { render } from "@testing-library/react"
import { vi } from "vitest"

import { AppSidebar } from "@/components/app-sidebar"
import { CommandMenu } from "@/components/command-menu"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"
import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"
import { WorkspaceProvider } from "@/components/workspace-provider"
import { api } from "@/lib/api"
import type { ChatResponse, DocumentSummary } from "@/lib/types"

export function doc(overrides: Partial<DocumentSummary> = {}): DocumentSummary {
  return {
    document_id: "0123456789abcdef",
    source_name: "Annual Report.pdf",
    status: "ready",
    chunks: 42,
    pages_with_chunks: 12,
    ingestion: null,
    ...overrides,
  }
}

export const answer = (overrides: Partial<ChatResponse> = {}): ChatResponse => ({
  answer: "Revenue reached 120 in 2025 [Annual Report.pdf, Page 4], up from 100 [Annual Report.pdf, Page 4].",
  abstained: false,
  citations: [{ document: "Annual Report.pdf", page: 4 }],
  sources: [
    { source_name: "Annual Report.pdf", page_number: 4, section_path: ["Financials"], content_type: "table", text: "|Year|Revenue|\n|2025|120|" },
    { source_name: "Annual Report.pdf", page_number: 4, section_path: ["Financials"], content_type: "text", text: "Revenue grew." },
  ],
  ...overrides,
})

/** Stub every API call; tests override the ones they exercise. */
export function mockApi(documents: DocumentSummary[] = [doc()]) {
  return {
    documents: vi.spyOn(api, "documents").mockResolvedValue(documents),
    health: vi.spyOn(api, "health").mockResolvedValue({ status: "ok" }),
    chat: vi.spyOn(api, "chat").mockResolvedValue(answer()),
    upload: vi.spyOn(api, "upload").mockResolvedValue({ source_name: "new.pdf", status: "queued" }),
    deleteDocument: vi.spyOn(api, "deleteDocument").mockResolvedValue({}),
  }
}

export function renderWorkspace(page: React.ReactNode) {
  return render(
    <TooltipProvider>
      <WorkspaceProvider>
        <SidebarProvider>
          <AppSidebar />
          <SidebarInset>{page}</SidebarInset>
          <CommandMenu />
        </SidebarProvider>
      </WorkspaceProvider>
      <Toaster />
    </TooltipProvider>,
  )
}
