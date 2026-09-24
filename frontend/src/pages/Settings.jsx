import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Page, PageHeader } from '../components/AppShell'
import { useTheme } from '../components/AppShell'
import { Badge, Card, ConnectionBadge, ErrorMessage, Field, Loading, Notice, Spinner } from '../components/ui'
import { useCan, useIsOwner, useUser } from '../user'
import {
  clearMeetstreamApiKey,
  completeSetup,
  exportWorkspaceUrl,
  getAgentCredentials,
  getProviderCatalog,
  getSetupStatus,
  getTunnel,
  getWriteTools,
  resetSetup,
  setMeetstreamApiKey,
  setTunnel,
  setWriteTools,
  testLlmProvider,
} from '../api'
import DatabasePicker, { isDatabaseFormComplete } from '../components/DatabasePicker'
import ProviderFields from '../components/ProviderFields'

const SECTIONS = [
  { id: 'ai', label: 'AI provider' },
  { id: 'database', label: 'Database' },
  { id: 'meetings', label: 'Meetings' },
  { id: 'general', label: 'General' },
]

function EnvManagedNotice() {
  return (
    <p
      className="mb-4 rounded-lg px-3 py-2 text-xs"
      style={{ backgroundColor: 'var(--color-peach-100)', color: 'var(--color-peach-700)' }}
    >
      Some of these values are set by environment variables on this deployment and cannot be
      changed here. Update your environment and restart to change them.
    </p>
  )
}

const TUNNEL_STATE = {
  off: 'Off',
  starting: 'Starting the tunnel…',
  verifying: 'Checking the address answers…',
  running: 'Running',
  error: 'Not running',
}

/**
 * Where MeetStream reaches this computer: a tunnel the app runs itself, or an
 * address someone set up. The agent can only look anything up in a call when
 * one of them works.
 */
function PublicAddress({ status, onStatus, onError }) {
  const saved = status.meetstream?.public_url || ''
  const [publicUrl, setPublicUrl] = useState(() => (saved.startsWith('https://') ? saved.replace(/\/mcp$/, '') : ''))
  const [busy, setBusy] = useState(false)
  const [tunnel, setTunnelState] = useState(status.meetstream?.tunnel || null)
  const envManaged = Boolean(status.environment_managed?.['meetstream.public_url'])
  const settling = tunnel?.enabled && ['starting', 'verifying'].includes(tunnel.state)

  // Follow the tunnel while it comes up; it reports within a few seconds.
  useEffect(() => {
    if (!settling) return undefined
    const timer = setInterval(async () => {
      try {
        const fresh = await getTunnel()
        setTunnelState(fresh)
        if (!['starting', 'verifying'].includes(fresh.state)) {
          onStatus((current) => ({
            ...current,
            meetstream: { ...current.meetstream, tunnel: fresh, public_url: fresh.public_url, public_url_problem: fresh.public_url_problem },
          }))
        }
      } catch {
        // The next tick retries.
      }
    }, 1500)
    return () => clearInterval(timer)
  }, [settling, onStatus])

  async function switchTunnel(enabled) {
    setBusy(true)
    onError(null)
    try {
      setTunnelState(await setTunnel(enabled))
    } catch (err) {
      onError(err.message)
    } finally {
      setBusy(false)
    }
  }

  async function saveAddress() {
    setBusy(true)
    onError(null)
    try {
      const updated = await completeSetup({ meetstream: { public_url: publicUrl.trim() } })
      onStatus(updated)
      // Show what was stored ("abc.trycloudflare.com" -> https://…).
      const stored = updated.meetstream?.public_url || ''
      setPublicUrl(stored.startsWith('https://') ? stored.replace(/\/mcp$/, '') : '')
    } catch (err) {
      onError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const usingTunnel = Boolean(tunnel?.enabled)

  return (
    <>
      <h3 className="mb-1 text-sm font-semibold">Public address</h3>
      <p className="mb-4 text-sm" style={{ color: 'var(--text-muted)' }}>
        How MeetStream reaches this computer during a call: the agent's memory lookups and the call's live
        updates are sent here. Without it the bot still joins, records and is summarised afterwards, but the
        agent cannot look anything up.
      </p>

      <div className="mb-5 rounded-xl p-4" style={{ border: '1px solid var(--border-subtle)' }}>
        <label className="flex cursor-pointer items-start gap-3">
          <input
            type="checkbox"
            className="mt-1"
            checked={usingTunnel}
            disabled={busy || envManaged || (tunnel && !tunnel.available && !usingTunnel)}
            onChange={(event) => switchTunnel(event.target.checked)}
          />
          <span>
            <span className="block text-sm font-medium" style={{ color: 'var(--text-strong)' }}>
              Start a tunnel automatically
            </span>
            <span className="block text-xs" style={{ color: 'var(--text-muted)' }}>
              While Meet Companion is open it runs a Cloudflare tunnel and uses its address — nothing to install
              or paste. Only MeetStream's traffic can use it: sign-in, your meetings and the rest of the app stay
              on this computer.
            </span>
          </span>
        </label>

        {tunnel && !tunnel.available && (
          <p className="mt-3 text-xs" style={{ color: 'var(--text-muted)' }}>
            This build does not include cloudflared. Install it (<code>winget install Cloudflare.cloudflared</code>,
            <code> brew install cloudflared</code>), or enter an address below.
          </p>
        )}

        {usingTunnel && (
          <div className="mt-3 text-xs">
            {tunnel.state === 'running' ? (
              <p style={{ color: 'var(--text-muted)' }}>
                <Badge tone="success">Running</Badge>{' '}
                at {tunnel.url} — the agent can look things up in a call. The address changes whenever the
                tunnel restarts; bots you launch always get the current one.
              </p>
            ) : tunnel.state === 'error' ? (
              <Notice title="The tunnel is not running">{tunnel.error}</Notice>
            ) : (
              <p className="flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
                <Spinner size={13} /> {TUNNEL_STATE[tunnel.state] || tunnel.state}
              </p>
            )}
          </div>
        )}
      </div>

      {!usingTunnel && (
        <>
          <Field
            label="Or use your own address"
            htmlFor="ms-public-url"
            hint={envManaged
              ? 'Set by the MCP_SERVER_URL environment variable on this machine.'
              : 'A tunnel you run (a named Cloudflare tunnel or an ngrok static domain keeps its address) or a real domain.'}
          >
            <input
              id="ms-public-url"
              className="mc-input"
              value={publicUrl}
              onChange={(event) => setPublicUrl(event.target.value)}
              placeholder="https://meet.example.com"
              autoComplete="off"
              spellCheck={false}
              disabled={envManaged}
            />
          </Field>
          <div className="mb-4">
            {status.meetstream?.public_url_problem ? (
              <Notice title="MeetStream can't reach this server">{status.meetstream.public_url_problem}</Notice>
            ) : (
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Reachable at {status.meetstream?.public_url} — the agent can look things up in a call.
              </p>
            )}
          </div>
          <button type="button" className="mc-btn mc-btn-primary" disabled={busy || envManaged} onClick={saveAddress}>
            {busy ? <Spinner size={14} /> : null} Save and check
          </button>
          <p className="mt-2 text-xs" style={{ color: 'var(--text-faint)' }}>
            Agents pick up a new address the next time you launch a bot — nothing to re-activate.
          </p>
        </>
      )}
    </>
  )
}

export default function Settings() {
  const user = useUser()
  const isOwner = useIsOwner()
  const canExport = useCan('export_workspace')
  // ?section=meetings opens straight on a section: the agent's "can't look
  // anything up" warning links here.
  const [searchParams] = useSearchParams()
  const [section, setSection] = useState(() => searchParams.get('section') || 'ai')
  const [status, setStatus] = useState(null)
  const [catalog, setCatalog] = useState(null)
  const [error, setError] = useState(null)

  const [provider, setProvider] = useState('')
  const [values, setValues] = useState({})
  const [testResult, setTestResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState(false)

  const [meetstreamKey, setMeetstreamKey] = useState('')
  const [meetstreamKeyStatus, setMeetstreamKeyStatus] = useState(null)
  const [credentials, setCredentials] = useState(null)
  const [writeTools, setWriteToolsState] = useState(null)
  const [webhookSecret, setWebhookSecret] = useState('')

  const [dbProvider, setDbProvider] = useState('')
  const [dbValues, setDbValues] = useState({})
  const [dbTest, setDbTest] = useState(null)
  const [dbSaved, setDbSaved] = useState(false)
  const [dbBusy, setDbBusy] = useState(false)
  const [theme, toggleTheme] = useTheme()

  async function load() {
    try {
      const [statusData, catalogData, credentialData, writeToolsData] = await Promise.all([
        getSetupStatus(),
        getProviderCatalog(),
        getAgentCredentials().catch(() => null),
        getWriteTools().catch(() => null),
      ])
      setStatus(statusData)
      setCatalog(catalogData)
      setCredentials(credentialData)
      setWriteToolsState(writeToolsData)
      setProvider(statusData.llm?.provider || 'ollama')
      setDbProvider(statusData.database?.provider || 'sqlite')

      setValues({
        model: statusData.llm?.model || '',
        base_url: statusData.llm?.base_url || '',
        api_key: '',
      })
    } catch (err) {
      setError(err.message)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const descriptor = useMemo(
    () => catalog?.llm.find((item) => item.name === provider) || null,
    [catalog, provider],
  )

  const envManaged = status?.environment_managed || {}
  const anyEnvManaged = Object.values(envManaged).some(Boolean)

  function buildPayload() {
    return {
      provider,
      model: values.model || null,
      base_url: values.base_url || null,
      // An empty box means "leave the stored key alone" - the real key is
      // never sent to the browser, so it cannot be echoed back.
      api_key: values.api_key ? values.api_key : null,
    }
  }

  async function save() {
    setBusy(true)
    setSaved(false)
    setError(null)
    try {
      const updated = await completeSetup({ llm: buildPayload() })
      setStatus(updated)
      setValues((current) => ({ ...current, api_key: '' }))
      setSaved(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  async function runTest() {
    setBusy(true)
    setTestResult(null)
    try {
      setTestResult(await testLlmProvider(buildPayload()))
    } catch (err) {
      setTestResult({ ok: false, detail: err.message })
    } finally {
      setBusy(false)
    }
  }

  async function saveMeetstreamKey() {
    setBusy(true)
    setMeetstreamKeyStatus(null)
    try {
      const result = await setMeetstreamApiKey(meetstreamKey.trim())
      setMeetstreamKey('')
      const tunnelNote = result.tunnel_started
        ? ' Also started a tunnel so the agent can reach this computer during calls — see Public address below.'
        : ''
      setMeetstreamKeyStatus(
        result.connected
          ? { ok: true, detail: `Connected to MeetStream.${tunnelNote}` }
          : { ok: false, detail: (result.connection_error || 'Saved, but could not verify the connection.') + tunnelNote },
      )
      await load()
    } catch (err) {
      setMeetstreamKeyStatus({ ok: false, detail: err.message })
    } finally {
      setBusy(false)
    }
  }

  async function handleReset() {
    if (!window.confirm('Clear the saved AI, database and MeetStream settings? Accounts, meetings and notes are kept; you will sign in and configure again from Settings.')) {
      return
    }
    await resetSetup().catch((err) => setError(err.message))
    window.location.reload()
  }

  if (error && !status) return <Page><ErrorMessage title="Could not load settings" detail={error} /></Page>
  if (!status || !catalog) return <Page><Loading /></Page>

  return (
    <Page>
      <PageHeader title="Settings" description="Configure the infrastructure Meet Companion runs on." />

      <div className="grid gap-5 lg:grid-cols-[13rem_1fr]">
        <nav className="flex gap-1 overflow-x-auto lg:flex-col">
          {SECTIONS.map((item) => (
            <button
              key={item.id}
              type="button"
              className="mc-nav-item whitespace-nowrap"
              aria-current={section === item.id ? 'page' : undefined}
              onClick={() => setSection(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="min-w-0">
          {error && (
            <div className="mb-4">
              <ErrorMessage title="Something went wrong" detail={error} />
            </div>
          )}

          {status.read_only && (section === 'ai' || section === 'database') && (
            <Card>
              <h2 className="mb-1 text-base font-semibold">Managed by a workspace owner</h2>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                The AI provider ({status.llm?.provider || 'not set'}
                {status.llm?.model ? ` · ${status.llm.model}` : ''}) and database ({status.database?.provider}) are
                shared by everyone on this server. Ask an owner of your workspace to change them.
              </p>
            </Card>
          )}

          {section === 'ai' && user?.workspace_ai?.mode === 'workspace' && (
            <Card className="mb-4">
              <h2 className="mb-1 text-base font-semibold">This workspace uses one provider for everyone</h2>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                An owner set {user.workspace_ai.provider}
                {user.workspace_ai.model ? ` (${user.workspace_ai.model})` : ''} for the whole workspace, so the
                settings below are not used here. They still apply to your other workspaces.
              </p>
            </Card>
          )}

          {section === 'ai' && !status.read_only && (
            <Card>
              <h2 className="mb-1 text-base font-semibold">AI provider</h2>
              <p className="mb-5 text-sm" style={{ color: 'var(--text-muted)' }}>
                Used for memory extraction and Ask AI.
              </p>

              {anyEnvManaged && <EnvManagedNotice />}

              <Field label="Provider" htmlFor="provider">
                <select
                  id="provider"
                  className="mc-input"
                  value={provider}
                  onChange={(event) => {
                    const next = catalog.llm.find((item) => item.name === event.target.value)
                    setProvider(event.target.value)
                    setTestResult(null)
                    setSaved(false)
                    // Start from the new provider's defaults - keeping the old
                    // one's host or key would send Groq requests to Ollama.
                    const defaults = {}
                    next?.fields.forEach((field) => {
                      if (field.default !== null && field.default !== undefined) defaults[field.key] = field.default
                    })
                    setValues({ model: '', base_url: '', api_key: '', ...defaults })
                  }}
                  disabled={envManaged['llm.provider']}
                >
                  {catalog.llm.map((item) => (
                    <option key={item.name} value={item.name}>{item.label}</option>
                  ))}
                </select>
              </Field>

              {descriptor && (
                <ProviderFields
                  descriptor={descriptor}
                  values={values}
                  onChange={(key, value) => {
                    setValues((current) => ({ ...current, [key]: value }))
                    setSaved(false)
                  }}
                  discoveredModels={testResult?.models || []}
                  keyHint={
                    status.llm?.api_key && status.llm?.provider === provider
                      ? `Currently set (${status.llm.api_key}). Leave blank to keep it.`
                      : undefined
                  }
                />
              )}

              <div className="flex flex-wrap items-center gap-2">
                <button type="button" className="mc-btn mc-btn-primary" onClick={save} disabled={busy}>
                  {busy ? <Spinner size={14} /> : null} Save changes
                </button>
                <button type="button" className="mc-btn mc-btn-secondary" onClick={runTest} disabled={busy}>
                  Test connection
                </button>
                {saved && <span className="text-sm" style={{ color: 'var(--color-brand-600)' }}>Saved.</span>}
                {testResult && (
                  <span
                    className="text-sm"
                    style={{ color: testResult.ok ? 'var(--color-brand-600)' : 'var(--color-rose-700)' }}
                  >
                    {testResult.ok ? 'Connected.' : testResult.detail}
                  </span>
                )}
              </div>
            </Card>
          )}

          {section === 'database' && !status.read_only && (
            <Card>
              <h2 className="mb-1 text-base font-semibold">Database</h2>
              <p className="mb-4 text-sm" style={{ color: 'var(--text-muted)' }}>
                Where meetings, notes and memory are stored. Pick any provider you like.
              </p>

              <div className="mb-5 flex flex-wrap items-center gap-2 text-sm">
                <span style={{ color: 'var(--text-muted)' }}>Currently using</span>
                <Badge tone="brand">
                  {catalog.databases.find((d) => d.name === status.database?.provider)?.label || status.database?.dialect}
                </Badge>
                <ConnectionBadge connected={status.database?.connected} />
                {status.database?.url && (
                  <code className="rounded px-1.5 py-0.5 text-xs" style={{ backgroundColor: 'var(--surface-raised)' }}>
                    {status.database.url}
                  </code>
                )}
              </div>
              {status.database?.connected === false && (
                <div className="mb-5">
                  <ErrorMessage title="The database is not answering" detail={status.database.error} onRetry={load} />
                </div>
              )}

              {status.environment_managed?.['database.url'] ? (
                <EnvManagedNotice />
              ) : (
                <>
                  <DatabasePicker
                    catalog={catalog.databases}
                    provider={dbProvider}
                    values={dbValues}
                    current={status.database?.provider}
                    connected={status.database?.connected}
                    onProviderChange={(name) => {
                      setDbProvider(name)
                      setDbSaved(false)
                    }}
                    onValuesChange={(next) => {
                      setDbValues(next)
                      setDbSaved(false)
                    }}
                    testResult={dbTest}
                    onTestResult={setDbTest}
                  />
                  <div className="mt-5 flex flex-wrap items-center gap-3">
                    <button
                      type="button"
                      className="mc-btn mc-btn-primary"
                      disabled={dbBusy || !isDatabaseFormComplete(catalog.databases.find((d) => d.name === dbProvider), dbValues)}
                      onClick={async () => {
                        setDbBusy(true)
                        setError(null)
                        try {
                          setStatus(await completeSetup({ database: { provider: dbProvider, values: dbValues } }))
                          setDbSaved(true)
                          // An empty database gets this account recreated in
                          // it (same email and password) and a fresh session;
                          // one that already has people in it shows sign-in.
                          setTimeout(() => window.location.reload(), 1600)
                        } catch (err) {
                          setError(err.message)
                        } finally {
                          setDbBusy(false)
                        }
                      }}
                    >
                      {dbBusy ? <Spinner size={14} /> : null} Save database
                    </button>
                    {dbSaved && (
                      <span className="text-sm" style={{ color: 'var(--color-brand-600)' }}>
                        Switched. Your account was carried over; meetings and notes stay in the previous database — reloading…
                      </span>
                    )}
                  </div>
                </>
              )}
            </Card>
          )}

          {section === 'meetings' && (
            <Card>
              <h2 className="mb-1 text-base font-semibold">Meeting capture</h2>
              <p className="mb-5 text-sm" style={{ color: 'var(--text-muted)' }}>
                Meet Companion uses MeetStream to send a bot into calls and produce transcripts.
                Without a key you can still use notes, search and Ask AI.
              </p>

              <Field
                label="Your MeetStream API key"
                hint={
                  credentials?.meetstream_api_key?.configured
                    ? `Currently set (${credentials.meetstream_api_key.masked_value}). Bots you launch use this key. Enter a new key to replace it.`
                    : 'Personal to you: bots you launch are created and billed under this MeetStream account.'
                }
                htmlFor="ms-key"
              >
                <input
                  id="ms-key"
                  className="mc-input"
                  type="password"
                  value={meetstreamKey}
                  onChange={(event) => {
                    setMeetstreamKey(event.target.value)
                    setMeetstreamKeyStatus(null)
                  }}
                  autoComplete="off"
                />
              </Field>

              {meetstreamKeyStatus && (
                <p
                  className="mb-3 text-sm"
                  style={{ color: meetstreamKeyStatus.ok ? 'var(--success-text, #2f9e5b)' : 'var(--danger-text, #c0392b)' }}
                >
                  {meetstreamKeyStatus.ok ? '✅' : '⚠️'} {meetstreamKeyStatus.detail}
                </p>
              )}

              <div className="flex gap-2">
                <button
                  type="button"
                  className="mc-btn mc-btn-primary"
                  onClick={saveMeetstreamKey}
                  disabled={busy || !meetstreamKey.trim()}
                >
                  Save key
                </button>
                {credentials?.meetstream_api_key?.is_personal && (
                  <button
                    type="button"
                    className="mc-btn mc-btn-danger"
                    onClick={async () => {
                      await clearMeetstreamApiKey().catch(() => {})
                      await load()
                    }}
                  >
                    Remove key
                  </button>
                )}
              </div>

              {isOwner && writeTools && (
                <>
                  <hr className="my-6" style={{ borderColor: 'var(--border-subtle)' }} />
                  <h3 className="mb-1 text-sm font-semibold">In-call agent permissions</h3>
                  <p className="mb-3 text-sm" style={{ color: 'var(--text-muted)' }}>
                    The agent always reads meeting memory. With write tools on it can also add notes and
                    create or update action items from what is said in a call — anyone in the meeting
                    can trigger that. Turn it off to keep the agent read-only.
                  </p>
                  <label className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={writeTools.enabled}
                      disabled={busy}
                      onChange={async (event) => {
                        setBusy(true)
                        try {
                          setWriteToolsState(await setWriteTools(event.target.checked))
                        } catch (err) {
                          setError(err.message)
                        } finally {
                          setBusy(false)
                        }
                      }}
                    />
                    Allow the agent to write notes and action items
                  </label>
                </>
              )}

              {isOwner && (
                <>
                  <hr className="my-6" style={{ borderColor: 'var(--border-subtle)' }} />
                  <PublicAddress status={status} onStatus={setStatus} onError={setError} />

                  <hr className="my-6" style={{ borderColor: 'var(--border-subtle)' }} />
                  <h3 className="mb-1 text-sm font-semibold">Webhook signing secret</h3>
                  <p className="mb-4 text-sm" style={{ color: 'var(--text-muted)' }}>
                    MeetStream signs every webhook it sends with this secret, and the server rejects
                    deliveries that do not match. Set the same value in your MeetStream webhook
                    settings.{' '}
                    {status.meetstream?.webhook_secret_configured ? 'A secret is currently set.' : 'No secret is set yet — unsigned deliveries are accepted only for bots this server launched.'}
                  </p>
                  <Field label="Webhook secret" htmlFor="ms-webhook">
                    <input
                      id="ms-webhook"
                      className="mc-input"
                      type="password"
                      value={webhookSecret}
                      onChange={(event) => setWebhookSecret(event.target.value)}
                      autoComplete="off"
                      disabled={status.environment_managed?.['meetstream.webhook_secret']}
                    />
                  </Field>
                  <button
                    type="button"
                    className="mc-btn mc-btn-primary"
                    disabled={busy || !webhookSecret.trim()}
                    onClick={async () => {
                      setBusy(true)
                      try {
                        await completeSetup({ meetstream: { webhook_secret: webhookSecret.trim() } })
                        setWebhookSecret('')
                        await load()
                      } catch (err) {
                        setError(err.message)
                      } finally {
                        setBusy(false)
                      }
                    }}
                  >
                    Save secret
                  </button>
                </>
              )}
            </Card>
          )}

          {section === 'general' && (
            <div className="flex flex-col gap-4">
              <Card>
                <h2 className="mb-1 text-base font-semibold">Appearance</h2>
                <p className="mb-4 text-sm" style={{ color: 'var(--text-muted)' }}>
                  Meet Companion follows your system theme by default.
                </p>
                <button type="button" className="mc-btn mc-btn-secondary" onClick={toggleTheme}>
                  Switch to {theme === 'dark' ? 'light' : 'dark'} theme
                </button>
              </Card>

              {canExport && (
              <Card>
                <h2 className="mb-1 text-base font-semibold">Export your data</h2>
                <p className="mb-4 text-sm" style={{ color: 'var(--text-muted)' }}>
                  Everything in this workspace, in a format that outlives the app. Markdown is a
                  zip you can open as a vault (folders preserved); JSON keeps ids and metadata.
                </p>
                <div className="flex flex-wrap gap-2">
                  <a className="mc-btn mc-btn-secondary" href={exportWorkspaceUrl('md')} download>
                    Download Markdown (.zip)
                  </a>
                  <a className="mc-btn mc-btn-secondary" href={exportWorkspaceUrl('json')} download>
                    Download JSON
                  </a>
                </div>
              </Card>
              )}

              {!status.read_only && (
              <Card>
                <h2 className="mb-1 text-base font-semibold">Data management</h2>
                <p className="mb-4 text-sm" style={{ color: 'var(--text-muted)' }}>
                  Clears the saved AI provider, database and MeetStream settings. Accounts, meetings
                  and notes are kept, so this does not reopen the first-run wizard — you sign in and
                  configure again from here. For a truly fresh start, delete the data directory.
                </p>
                <button type="button" className="mc-btn mc-btn-danger" onClick={handleReset}>
                  Reset configuration
                </button>
              </Card>
              )}
            </div>
          )}
        </div>
      </div>
    </Page>
  )
}
