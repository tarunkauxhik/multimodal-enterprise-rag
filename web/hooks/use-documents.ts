"use client"

import { useCallback, useEffect, useRef, useState } from "react"

import { notify } from "@/components/notify"
import { api, ApiError, errorMessage, MAX_UPLOAD_MB } from "@/lib/api"
import { isActive, sortDocuments } from "@/lib/status"
import type { DocumentSummary } from "@/lib/types"

export const POLL_MS = 2000
const QUEUE_CAPACITY = 2 // api.py: one running ingestion plus one waiting; a 429 still covers any mismatch

export interface UploadItem {
  id: string
  name: string
  state: "waiting" | "uploading"
  progress: number // 0..1 of the request body sent
}

export function validateFile(file: File): string | null {
  if (!file.name.toLowerCase().endsWith(".pdf") && file.type !== "application/pdf") return "Only PDF documents are supported."
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `This PDF is larger than the ${MAX_UPLOAD_MB} MB limit.`
  if (file.size === 0) return "This file is empty."
  return null
}

/** Workspace documents: loads the list, polls only while something is queued, processing or
 * waiting to upload, and sends uploads one at a time within the API's queue capacity. */
export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentSummary[] | null>(null) // null until first load
  const [error, setError] = useState<ApiError | null>(null)
  const [uploads, setUploads] = useState<UploadItem[]>([])
  const files = useRef(new Map<string, File>())
  const sending = useRef(false)
  const queueFull = useRef(false) // after a 429, wait for a fresh list before trying again
  const lastStatus = useRef(new Map<string, DocumentSummary["status"]>())

  const refresh = useCallback(async () => {
    try {
      const docs = await api.documents()
      queueFull.current = false
      announceFinished(lastStatus.current, docs)
      lastStatus.current = new Map(docs.map((d) => [d.document_id, d.status]))
      setDocuments(sortDocuments(docs))
      setError(null)
      return docs
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(-1))
      return null
    }
  }, [])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial load of external data
    void refresh()
    const onFocus = () => void refresh()
    window.addEventListener("focus", onFocus)
    return () => window.removeEventListener("focus", onFocus)
  }, [refresh])

  const polling = !error && ((documents ?? []).some(isActive) || uploads.length > 0)
  useEffect(() => {
    if (!polling) return
    const timer = window.setInterval(() => void refresh(), POLL_MS)
    return () => window.clearInterval(timer)
  }, [polling, refresh])

  const update = (id: string, patch: Partial<UploadItem>) =>
    setUploads((items) => items.map((item) => (item.id === id ? { ...item, ...patch } : item)))
  const drop = (id: string) => {
    files.current.delete(id)
    setUploads((items) => items.filter((item) => item.id !== id))
  }

  // Send the next waiting file when the API has room. Runs after every list refresh.
  useEffect(() => {
    const next = uploads.find((item) => item.state === "waiting")
    const busy = (documents ?? []).filter(isActive).length
    if (!next || sending.current || queueFull.current || documents === null || busy >= QUEUE_CAPACITY) return
    const file = files.current.get(next.id)!
    sending.current = true
    update(next.id, { state: "uploading", progress: 0 })
    api
      .upload(file, (progress) => update(next.id, { progress }))
      .then(async (accepted) => {
        if (accepted.status === "ready") notify("📎", `${accepted.source_name} is already in the workspace.`)
        else notify("📄", `${accepted.source_name} was added and is being processed.`)
        await refresh() // the document now appears as queued
        drop(next.id)
      })
      .catch((err) => {
        if (err instanceof ApiError && err.status === 429) {
          queueFull.current = true
          update(next.id, { state: "waiting", progress: 0 }) // retried after the next refresh
        } else {
          notify("⚠️", `${next.name}: ${errorMessage(err, "upload")}`, "error")
          drop(next.id)
        }
      })
      .finally(() => {
        sending.current = false
      })
  }, [uploads, documents, refresh])

  const addFiles = useCallback((list: Iterable<File>) => {
    const accepted: UploadItem[] = []
    for (const file of list) {
      const problem = validateFile(file)
      if (problem) {
        notify("⚠️", `${file.name}: ${problem}`, "error")
        continue
      }
      const id = crypto.randomUUID()
      files.current.set(id, file)
      accepted.push({ id, name: file.name, state: "waiting", progress: 0 })
    }
    if (accepted.length) setUploads((items) => [...items, ...accepted])
  }, [])

  const remove = useCallback(
    async (doc: DocumentSummary) => {
      try {
        await api.deleteDocument(doc.document_id)
        notify("🗑️", `${doc.source_name} was deleted.`)
      } catch (err) {
        if (!(err instanceof ApiError && err.status === 404)) {
          notify("⚠️", errorMessage(err, "delete"), "error")
          return false
        }
      }
      await refresh()
      return true
    },
    [refresh],
  )

  return { documents, error, uploads, refresh, addFiles, remove }
}

// Tells the user when a document they were waiting for finishes, wherever they are in the app.
const FINISHED: Partial<Record<DocumentSummary["status"], [string, string, "default" | "error"]>> = {
  ready: ["✅", "is ready for questions.", "default"],
  failed: ["⚠️", "couldn't be processed.", "error"],
  incomplete: ["⚠️", "was only partially indexed. Upload it again to repair it.", "error"],
  empty: ["🫥", "has no readable text.", "default"],
}

function announceFinished(before: Map<string, DocumentSummary["status"]>, docs: DocumentSummary[]) {
  for (const doc of docs) {
    const was = before.get(doc.document_id)
    const done = FINISHED[doc.status]
    if ((was === "queued" || was === "processing") && done) notify(done[0], `${doc.source_name} ${done[1]}`, done[2])
  }
}
