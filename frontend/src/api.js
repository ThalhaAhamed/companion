// In local dev, vite.config.js proxies /api to localhost:8000 - that proxy
// doesn't exist in a production static build, so a deployed frontend needs
// the real backend origin baked in at build time via VITE_API_BASE_URL.
const BASE = `${import.meta.env.VITE_API_BASE_URL || ''}/api`

/**
 * Pull a readable message out of an error response.
 *
 * FastAPI returns {"detail": "..."} and sometimes nests another JSON payload
 * inside it. Showing the raw body meant users saw the HTTP envelope and
 * escaped JSON rather than the sentence the server actually wrote.
 */
function extractMessage(body, status, statusText) {
  if (!body) return `${status} ${statusText}`

  let detail
  try {
    const parsed = JSON.parse(body)
    detail = typeof parsed === 'string' ? parsed : parsed.detail
  } catch {
    return body.length > 300 ? `${body.slice(0, 300)}â€¦` : body
  }

  if (Array.isArray(detail)) {
    // Pydantic validation errors arrive as a list of field problems.
    const first = detail[0]
    return first?.msg ? `${(first.loc || []).slice(-1)[0] || 'Input'}: ${first.msg}` : JSON.stringify(detail)
  }
  if (typeof detail !== 'string') return `${status} ${statusText}`

  // A provider's own error body is often embedded as JSON inside detail.
  const nested = detail.match(/\{.*\}/s)
  if (nested) {
    try {
      const inner = JSON.parse(nested[0])
      const innerMessage = inner.error?.message || inner.error || inner.detail || inner.message
      if (typeof innerMessage === 'string') {
        return `${detail.slice(0, nested.index).trim()} ${innerMessage}`.trim()
      }
    } catch {
      // Not JSON after all - fall through to the detail as written.
    }
  }
  return detail
}

async function req(path, options) {
  const res = await fetch(`${BASE}${path}`, { credentials: 'include', ...options })
  // A 401 from the login endpoint is a wrong password, not a lost session -
  // let its own message through instead of kicking the user to sign-in.
  if (res.status === 401 && path !== '/auth/login') {
    window.dispatchEvent(new Event('hub:unauthorized'))
    throw new Error('Your session has expired. Please sign in again.')
  }
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    const error = new Error(extractMessage(body, res.status, res.statusText))
    error.status = res.status
    throw error
  }
  // 204 No Content (deletes) and other empty bodies have nothing to parse.
  if (res.status === 204) return null
  const text = await res.text()
  return text ? JSON.parse(text) : null
}

export function checkAuth() {
  return req('/auth/check')
}

export function login(email, password) {
  return req('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
}

export function logout() {
  return req('/auth/logout', { method: 'POST' })
}

export function listMembers() {
  return req('/members')
}

export function addMember({ name, email, password, workspace_name, join_code }) {
  return req('/members', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password, workspace_name, join_code }),
  })
}

export function getWorkspace() {
  return req('/members/workspace')
}

export function setWorkspaceAI(config) {
  return req('/members/workspace/ai', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
}

export function testWorkspaceAI(config) {
  return req('/members/workspace/ai/test', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
}

export function renameWorkspace(name) {
  return req('/members/workspace', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
}

export function regenerateJoinCode() {
  return req('/members/workspace/join-code', { method: 'POST' })
}

export function deleteWorkspace() {
  return req('/members/workspace', { method: 'DELETE' })
}

export function setWorkspacePermissions(member_permissions) {
  return req('/members/workspace/permissions', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ member_permissions }),
  })
}

export function approveMember(id) {
  return req(`/members/${id}/approve`, { method: 'POST' })
}

export function removeMember(id) {
  return req(`/members/${id}`, { method: 'DELETE' })
}

export function resetMemberPassword(id, new_password) {
  return req(`/members/${id}/reset-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ new_password }),
  })
}

export function setMemberRole(id, role) {
  return req(`/members/${id}/role`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ role }),
  })
}

export function updateSelf(patch) {
  return req('/members/me', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
}

export function listMeetings(day) {
  const qs = day ? `?day=${day}&limit=100` : '?limit=100'
  return req(`/meetings${qs}`)
}

export function getMeeting(id) {
  return req(`/meetings/${id}`)
}

export function getTranscript(id) {
  return req(`/meetings/${id}/transcript`)
}

export function listDocuments(day) {
  const qs = day ? `?day=${day}` : ''
  return req(`/documents${qs}`)
}

export function createMeeting({ meeting_url, title }) {
  return req('/meetings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ meeting_url, title: title || undefined }),
  })
}

export function getMeetingBot(id) {
  return req(`/meetings/${id}/bot`)
}

export function getKnowledgeGraph() {
  return req('/graph')
}

export function listImportableBots({ from, to } = {}) {
  const query = new URLSearchParams()
  if (from) query.set('from', from)
  if (to) query.set('to', to)
  const qs = query.toString()
  return req(`/meetings/importable${qs ? `?${qs}` : ''}`)
}

export function importBot({ bot_id, title, platform, meeting_url }) {
  return req('/meetings/import', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bot_id, title, platform, meeting_url }),
  })
}

export function getAgent(agentConfigId) {
  return req(agentConfigId ? `/agent?agent_config_id=${encodeURIComponent(agentConfigId)}` : '/agent')
}

export function updateAgent(patch) {
  return req('/agent', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
}

export function deleteMeeting(id) {
  return req(`/meetings/${id}`, { method: 'DELETE' })
}

export function stopMeetingBot(id) {
  return req(`/meetings/${id}/stop`, { method: 'POST' })
}

export function searchMemory(query, opts = {}) {
  return req('/search/memory', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, ...opts }),
  })
}

export function updateActionItem(id, patch) {
  return req(`/action-items/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
}

export function getWriteTools() {
  return req('/agent/write-tools')
}

export function setWriteTools(enabled) {
  return req('/agent/write-tools', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  })
}

export function getAgentCredentials() {
  return req('/agent/credentials')
}

export function setMeetstreamApiKey(meetstream_api_key) {
  return req('/agent/api-key', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ meetstream_api_key }),
  })
}

export function clearMeetstreamApiKey() {
  return req('/agent/api-key', { method: 'DELETE' })
}

export function listAgents() {
  return req('/agent/list')
}

export function listImportableAgents() {
  return req('/agent/importable')
}

export function createAgent(payload) {
  return req('/agent', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function setAgentMode(agent_config_id, mode) {
  return req('/agent/mode', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ agent_config_id, mode }),
  })
}

export function deleteAgent(agent_config_id) {
  return req(`/agent?agent_config_id=${encodeURIComponent(agent_config_id)}`, { method: 'DELETE' })
}

export function activateAgent(agent_config_id) {
  return req('/agent/activate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ agent_config_id }),
  })
}

export function deleteDocument(id) {
  return req(`/documents/${id}`, { method: 'DELETE' })
}

export function uploadDocument(file) {
  const form = new FormData()
  form.append('file', file)
  return req('/documents/upload', { method: 'POST', body: form })
}

// ---------------------------------------------------------------------------
// Setup and configuration
// ---------------------------------------------------------------------------

export function getSetupStatus() {
  return req('/setup/status')
}

export function getProviderCatalog() {
  return req('/setup/providers')
}

export function testLlmProvider(config) {
  return req('/setup/test-llm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
}

export function testDatabase(payload) {
  // payload: { provider, values } or { url }
  return req('/setup/test-database', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

/** The automatic tunnel: state, address, and whether MeetStream can reach it. */
export function getTunnel() {
  return req('/setup/tunnel')
}

export function setTunnel(enabled) {
  return req('/setup/tunnel', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled }) })
}

export function completeSetup(payload) {
  return req('/setup/complete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function resetSetup() {
  return req('/setup/reset', { method: 'POST' })
}

// ---------------------------------------------------------------------------
// Notebook
// ---------------------------------------------------------------------------

export function listFolders() {
  return req('/notebook/folders')
}

export function createFolder({ name, parent_id = null }) {
  return req('/notebook/folders', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, parent_id }),
  })
}

export function updateFolder(id, patch) {
  return req(`/notebook/folders/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
}

export function deleteFolder(id, { cascade = false } = {}) {
  return req(`/notebook/folders/${id}?cascade=${cascade}`, { method: 'DELETE' })
}

export function listNotes(params = {}) {
  const query = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') query.set(key, value)
  })
  const suffix = query.toString()
  return req(`/notebook/notes${suffix ? `?${suffix}` : ''}`)
}

export function getNote(id) {
  return req(`/notebook/notes/${id}`)
}

export function createNote(payload) {
  return req('/notebook/notes', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function updateNote(id, patch) {
  return req(`/notebook/notes/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
}

export function deleteNote(id) {
  return req(`/notebook/notes/${id}`, { method: 'DELETE' })
}

export function listNoteTags() {
  return req('/notebook/tags')
}

export function askNotebook(payload) {
  return req('/notebook/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function listActionItems(params = {}) {
  const query = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') query.set(key, value)
  })
  const suffix = query.toString()
  return req(`/action-items${suffix ? `?${suffix}` : ''}`)
}

export function getAgentTemplate() {
  return req('/agent/template')
}

export function updateAgentTemplate(patch) {
  return req('/agent/template', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
}

export function syncMeetingNotes() {
  return req('/notebook/sync-meetings', { method: 'POST' })
}

export function uploadTranscript(payload) {
  return req('/meetings/upload', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function reprocessMeeting(id) {
  return req(`/meetings/${id}/reprocess`, { method: 'POST' })
}

/**
 * Download URLs for exports.
 *
 * These are plain links rather than fetch calls: the browser sends the session
 * cookie, honours the Content-Disposition filename the server sets, and streams
 * the file straight to disk without it passing through JS memory.
 */
export function exportNoteUrl(id, format = 'md') {
  return `${BASE}/export/note/${id}?format=${format}`
}

export function exportMeetingUrl(id, format = 'md') {
  return `${BASE}/export/meeting/${id}?format=${format}`
}

export function exportWorkspaceUrl(format = 'json') {
  return `${BASE}/export/workspace?format=${format}`
}

export function listMyWorkspaces() {
  return req('/members/workspaces')
}

export function activateWorkspace(id) {
  return req(`/members/workspaces/${id}/activate`, { method: 'POST' })
}

// Saved database connections (desktop only; 404 elsewhere).
export function listConnections() {
  return req('/connections')
}

export function addConnection(body) {
  return req('/connections', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
}

export function removeConnection(id) {
  return req(`/connections/${id}`, { method: 'DELETE' })
}

export function activateConnection(id, organization_id) {
  return req(`/connections/${id}/activate`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ organization_id: organization_id || null }),
  })
}

/** Does this code name a workspace on that database? Nothing is saved or switched. */
export function checkJoin({ url, provider, values, join_code }) {
  return req('/connections/check-join', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: url || undefined, provider, values, join_code }),
  })
}

export function joinWorkspace(joinCode) {
  return req('/members/workspaces/join', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ join_code: joinCode }),
  })
}

export function createWorkspace(name) {
  return req('/members/workspaces', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
}
