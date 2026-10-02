import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Page, PageHeader } from '../components/AppShell'
import { AskAiIcon } from '../components/Icons'
import { Badge, Card, EmptyState, ErrorMessage, Spinner } from '../components/ui'
import Markdown from '../components/Markdown'
import { askNotebook, deleteDocument, listDocuments, listFolders, uploadDocument } from '../api'
import { useCan } from '../user'

const SUGGESTIONS = [
  'What were the action items from my recent meetings?',
  'What decisions have we made about pricing?',
  'Summarise everything I have on Project Alpha.',
  'What questions are still unresolved?',
]

/**
 * Uploaded reference material - specs, contracts, handbooks - that Ask AI
 * reads alongside notes and meetings. Chunked and embedded on upload.
 */
function DocumentsPanel() {
  const canCreate = useCan('create_content')
  const canDelete = useCan('delete_content')
  const [documents, setDocuments] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const input = useRef(null)

  async function load() {
    try {
      setDocuments(await listDocuments())
    } catch (err) {
      setError(err.message)
    }
  }

  useEffect(() => {
    load()
  }, [])

  async function upload(event) {
    const file = event.target.files?.[0]
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      await uploadDocument(file)
      await load()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
      if (input.current) input.current.value = ''
    }
  }

  async function remove(doc) {
    if (!window.confirm(`Remove "${doc.filename}" from Ask AI?`)) return
    setError(null)
    try {
      await deleteDocument(doc.document_id)
      await load()
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <Card className="mb-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold">Documents</h2>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Reference material Ask AI can quote: PDF, Word, Markdown, text or CSV, up to 25 MB.
          </p>
        </div>
        {canCreate && (
          <label className="mc-btn mc-btn-secondary cursor-pointer">
            {busy ? <Spinner size={13} /> : null} {busy ? 'Indexing…' : 'Upload document'}
            <input ref={input} type="file" accept=".pdf,.docx,.txt,.md,.csv" className="hidden" onChange={upload} disabled={busy} />
          </label>
        )}
      </div>
      {error && <div className="mt-3"><ErrorMessage title="Document problem" detail={error} /></div>}
      {documents && documents.length > 0 && (
        <ul className="mt-3 flex flex-wrap gap-2">
          {documents.map((doc) => (
            <li key={doc.document_id || doc.filename} className="flex items-center gap-1">
              <Badge tone="neutral">
                {doc.filename} · {doc.chunks_count} chunk{doc.chunks_count === 1 ? '' : 's'}
              </Badge>
              {doc.document_id && canDelete && (
                <button
                  type="button"
                  className="text-xs"
                  style={{ color: 'var(--text-faint)' }}
                  onClick={() => remove(doc)}
                  aria-label={`Remove ${doc.filename}`}
                  title="Remove"
                >
                  ✕
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}

function flatten(nodes, depth = 0, acc = []) {
  for (const node of nodes || []) {
    acc.push({ id: node.id, name: `${'— '.repeat(depth)}${node.name}` })
    flatten(node.children, depth + 1, acc)
  }
  return acc
}

export default function AskAi() {
  const [question, setQuestion] = useState('')
  const [scope, setScope] = useState('all')
  const [folders, setFolders] = useState([])
  const [thread, setThread] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    listFolders()
      .then((data) => setFolders(flatten(data.folders)))
      .catch(() => {})
  }, [])

  async function ask(text) {
    const trimmed = (text ?? question).trim()
    if (!trimmed || busy) return

    // The conversation so far goes with the question, so a follow-up such as
    // "who owns it now?" is answered about what "it" was. Errors aren't turns.
    const history = thread
      .filter((entry) => entry.text)
      .slice(-6)
      .map((entry) => ({ role: entry.role, content: entry.text }))

    setBusy(true)
    setError(null)
    setQuestion('')
    setThread((current) => [...current, { role: 'user', text: trimmed }])

    try {
      const payload = { question: trimmed, history }
      if (scope === 'favorites') payload.favorites_only = true
      else if (scope !== 'all') payload.folder_id = scope

      const result = await askNotebook(payload)
      setThread((current) => [
        ...current,
        { role: 'assistant', text: result.answer, sources: result.sources, documents: result.documents, model: result.model },
      ])
    } catch (err) {
      setError(err.message)
      setThread((current) => current.slice(0, -1))
      setQuestion(trimmed)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Page>
      <PageHeader
        title="Ask AI"
        description="Questions answered from your notes, meetings and uploaded documents, using the provider you configured."
        actions={
          <select
            className="mc-input w-auto"
            value={scope}
            onChange={(event) => setScope(event.target.value)}
            aria-label="Limit to"
          >
            <option value="all">All notes</option>
            <option value="favorites">Favourites only</option>
            {folders.map((folder) => (
              <option key={folder.id} value={folder.id}>{folder.name}</option>
            ))}
          </select>
        }
      />

      {error && (
        <div className="mb-4">
          <ErrorMessage title="Could not answer that" detail={error} />
        </div>
      )}

      <DocumentsPanel />

      <Card padded={false} className="flex min-h-[60vh] flex-col">
        <div className="mc-scroll flex-1 overflow-y-auto p-5">
          {thread.length === 0 ? (
            <EmptyState
              icon={<AskAiIcon size={22} />}
              title="Ask anything about your notes"
              description="Answers are grounded in what you have captured — nothing is invented."
              action={
                <div className="flex flex-wrap justify-center gap-2">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      className="mc-btn mc-btn-secondary text-xs"
                      onClick={() => ask(suggestion)}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              }
            />
          ) : (
            <div className="flex flex-col gap-4">
              {thread.map((entry, index) => (
                <div
                  key={index}
                  className={entry.role === 'user' ? 'self-end max-w-[80%]' : 'max-w-[85%]'}
                >
                  <div
                    className={`rounded-xl px-4 py-3 text-sm leading-relaxed ${entry.role === 'user' ? 'whitespace-pre-wrap' : ''}`}
                    style={
                      entry.role === 'user'
                        ? { backgroundColor: 'var(--brand-solid)', color: 'var(--text-on-brand)' }
                        : { backgroundColor: 'var(--surface-raised)', color: 'var(--text-default)' }
                    }
                  >
                    {entry.role === 'user' ? entry.text : <Markdown source={entry.text} className="text-sm" />}
                  </div>

                  {(entry.sources?.length > 0 || entry.documents?.length > 0) && (
                    <div className="mt-2 flex flex-wrap items-center gap-1.5">
                      <span className="text-[0.7rem]" style={{ color: 'var(--text-faint)' }}>
                        Sources:
                      </span>
                      {entry.sources?.map((source) => (
                        // Only what the answer cited: a note, or a meeting it quoted from.
                        <Link
                          key={`${source.kind}-${source.id}`}
                          to={source.kind === 'meeting' ? `/meetings/${source.id}` : `/notebook/${source.id}`}
                          title={source.date ? `${source.kind === 'meeting' ? 'Meeting' : 'Note'} · ${source.date}` : undefined}
                        >
                          <Badge tone="brand">
                            {source.kind === 'meeting' ? '🎙 ' : ''}
                            {source.title}
                          </Badge>
                        </Link>
                      ))}
                      {entry.documents?.map((doc) => (
                        <Badge key={doc.id || doc.title} tone="neutral" title="Uploaded document">
                          📄 {doc.title}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {busy && (
                <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
                  <Spinner size={15} /> Thinking…
                </div>
              )}
            </div>
          )}
        </div>

        <form
          className="flex gap-2 p-4"
          style={{ borderTop: '1px solid var(--border-subtle)' }}
          onSubmit={(event) => {
            event.preventDefault()
            ask()
          }}
        >
          <input
            className="mc-input"
            placeholder="Ask about your notes…"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            aria-label="Your question"
          />
          <button type="submit" className="mc-btn mc-btn-primary" disabled={busy || !question.trim()}>
            {busy ? <Spinner size={14} /> : <AskAiIcon size={16} />}
            Ask
          </button>
        </form>
      </Card>
    </Page>
  )
}
