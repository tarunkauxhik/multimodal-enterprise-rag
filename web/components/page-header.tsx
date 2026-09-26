import { Separator } from "@/components/ui/separator"
import { SidebarTrigger } from "@/components/ui/sidebar"
import { cn } from "@/lib/utils"

/** titleVisible: the chat hides the title while the same question is visible as the page heading. */
export function PageHeader({ title, titleVisible = true, children }: { title: React.ReactNode; titleVisible?: boolean; children?: React.ReactNode }) {
  return (
    <header className="sticky top-0 z-10 flex h-12 shrink-0 items-center gap-2 bg-background/90 px-3 backdrop-blur supports-[backdrop-filter]:bg-background/75">
      <SidebarTrigger aria-label="Toggle sidebar" className="pointer-coarse:size-10" />
      <Separator orientation="vertical" className="mr-1 data-vertical:h-4 data-vertical:self-center" />
      <h1 className={cn("min-w-0 flex-1 truncate text-sm text-muted-foreground transition-opacity duration-200", !titleVisible && "opacity-0")}>{title}</h1>
      {children}
    </header>
  )
}
