import { Link } from 'react-router-dom'
import { useCallback, useEffect, useState } from 'react'
import { Page, PageHeader } from '../components/AppShell'
import { AskAiIcon, CheckIcon, PlusIcon, RobotIcon, TrashIcon } from '../components/Icons'
import { Badge, Card, EmptyState, ErrorMessage, Field, Loading, Modal, Notice, SectionCard, Spinner } from '../components/ui'
import {
  activateAgent,
  createAgent,
  deleteAgent,
  setAgentMode,
  getAgent,
  getAgentCredentials,
  getAgentTemplate,
  listAgents,
  listImportableAgents,
  updateAgent,
  updateAgentTemplate,
} from '../api'
import { useCan, useIsOwner, useUser } from '../user'

const MODES = ['realtime', 'pipeline']
const MODALITIES = ['text', 'audio', 'chat']

const EMPTY_FORM = {
  system_prompt: '',
  first_message: '',
  provider: '',
  model: '',
  voice: '',
  temperature: '',
  response_modality: '',
  tool_results_to_chat: false,
}

const BLANK_NEW_AGENT = {
  agent_name: '',
  activate: true,
  provider: '',
  model: '',
  voice: '',
  mode: 'realtime',
  system_prompt: '',
  first_message: '',
}

function NewAgentModal({ open, onClose, onCreated, template }) {
  const [form, setForm] = useState(BLANK_NEW_AGENT)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  // Every field starts as the template's value, so the modal only asks for
  // what is genuinely new (the name) and the rest can be tweaked or left.
  useEffect(() => {
    if (!open) return
    setForm({
      agent_name: '',
      activate: true,
      provider: template?.provider || '',
      model: template?.model || '',
      voice: template?.voice || '',
      mode: template?.mode || 'realtime',
      system_prompt: template?.system_prompt || '',
      first_message: template?.first_message || '',
    })
    setError(null)
  }, [open, template])

  function update(key, value) {
    setForm((current) => ({ ...current, [key]: value }))
  }

  async function submit(event) {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await createAgent(form)
      onCreated?.()
      onClose()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="New agent" width="34rem">
      <form onSubmit={submit}>
        <Field label="Agent name" hint="Participants address the agent by this name." htmlFor="agent-name">
          <input
            id="agent-name"
            className="mc-input"
            required
            value={form.agent_name}
            onChange={(event) => update('agent_name', event.target.value)}
          />
        </Field>

        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Provider" htmlFor="agent-provider">
            <input
              id="agent-provider"
              className="mc-input"
              value={form.provider}
              onChange={(event) => update('provider', event.target.value)}
            />
          </Field>
          <Field label="Model" htmlFor="agent-model">
            <input
              id="agent-model"
              className="mc-input"
              value={form.model}
              onChange={(event) => update('model', event.target.value)}
            />
          </Field>
          <Field label="Voice" htmlFor="agent-voice">
            <input
              id="agent-voice"
              className="mc-input"
              value={form.voice}
              onChange={(event) => update('voice', event.target.value)}
            />
          </Field>
          <Field label="Mode" htmlFor="agent-mode">
            <select
              id="agent-mode"
              className="mc-input"
              value={form.mode}
              onChange={(event) => update('mode', event.target.value)}
            >
              {MODES.map((mode) => (
                <option key={mode} value={mode}>{mode}</option>
              ))}
            </select>
          </Field>
        </div>

        <Field
          label="System prompt"
          hint="Copied from the template agent. {agent_name} is replaced with the name above."
          htmlFor="agent-prompt"
        >
          <textarea
            id="agent-prompt"
            className="mc-input min-h-32 font-mono text-xs"
            value={form.system_prompt || ''}
            onChange={(event) => update('system_prompt', event.target.value)}
          />
        </Field>

        <Field label="First message" hint="Posted into the meeting chat when the agent joins." htmlFor="agent-first">
          <textarea
            id="agent-first"
            className="mc-input min-h-20"
            value={form.first_message || ''}
            onChange={(event) => update('first_message', event.target.value)}
          />
        </Field>

        {error && <div className="mb-4"><ErrorMessage title="Could not create the agent" detail={error} /></div>}

        <button type="submit" className="mc-btn mc-btn-primary w-full" disabled={busy || !form.agent_name.trim()}>
          {busy ? <Spinner size={14} /> : null} Create agent
        </button>
      </form>
    </Modal>
  )
}

function ImportAgentsModal({ open, onClose, onImported }) {
  const [candidates, setCandidates] = useState(null)
  const [error, setError] = useState(null)
  const [busyId, setBusyId] = useState(null)

  useEffect(() => {
    if (!open) return
    setCandidates(null)
    setError(null)
    listImportableAgents()
      .then((data) => setCandidates(data.importable || []))
      .catch((err) => setError(err.message))
  }, [open])

  async function claim(agent) {
    setBusyId(agent.AgentConfigID)
    try {
      await activateAgent(agent.AgentConfigID)
      onImported?.()
      onClose()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusyId(null)
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="Import an agent from MeetStream" width="34rem">
      <p className="mb-4 text-sm" style={{ color: 'var(--text-muted)' }}>
        Agents that exist on your MeetStream account but that nobody here has claimed — created on
        MeetStream's own dashboard, or left behind by a removed member.
      </p>

      {error && <div className="mb-4"><ErrorMessage title="Import failed" detail={error} /></div>}

      {candidates === null && !error ? (
        <Loading />
      ) : candidates?.length === 0 ? (
        <EmptyState title="Nothing to import" description="Every agent on this account is already claimed." />
      ) : (
        <ul className="mc-scroll max-h-80 divide-y overflow-y-auto" style={{ borderColor: 'var(--border-subtle)' }}>
          {(candidates || []).map((agent) => (
            <li key={agent.AgentConfigID} className="flex items-center justify-between gap-3 py-3">
              <div className="min-w-0">
                <div className="truncate text-sm font-medium">{agent.AgentName || 'Untitled agent'}</div>
                <div className="mt-0.5 flex flex-wrap gap-1.5">
                  {agent.Mode && <Badge>{agent.Mode}</Badge>}
                  {agent.Model?.provider && <Badge>{agent.Model.provider}</Badge>}
                </div>
              </div>
              <button
                type="button"
                className="mc-btn mc-btn-secondary"
                onClick={() => claim(agent)}
                disabled={busyId === agent.AgentConfigID}
              >
                {busyId === agent.AgentConfigID ? <Spinner size={13} /> : null} Import
              </button>
            </li>
          ))}
        </ul>
      )}
    </Modal>
  )
}

const INTERACTION_MODES = [
  ['voice', 'Voice', 'Answers out loud when someone says its name.'],
  ['chat', 'Chat', 'Stays silent. When someone says its name, the answer is posted in the meeting chat.'],
]

/**
 * How this agent answers in a call - MeetStream's response_modality on the
 * agent, as a two-way choice. In both modes the agent hears the room and is
 * addressed by name: MeetStream does not deliver typed chat during a call.
 */
function InteractionModePicker({ agentConfigId, agentName, mode, readOnly, onChanged }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function choose(next) {
    if (next === mode || readOnly) return
    setBusy(true)
    setError(null)
    try {
      await setAgentMode(agentConfigId, next)
      onChanged?.(next)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mb-5">
      <div className="mc-label">How it answers in a call</div>
      <div className="grid gap-2 sm:grid-cols-2" role="radiogroup" aria-label="How the agent answers">
        {INTERACTION_MODES.map(([value, title, blurb]) => (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={mode === value}
            disabled={busy || readOnly}
            onClick={() => choose(value)}
            className="rounded-xl border p-3 text-left"
            style={{
              borderColor: mode === value ? 'var(--brand-ring)' : 'var(--border-subtle)',
              backgroundColor: mode === value ? 'var(--brand-soft)' : 'transparent',
              cursor: readOnly ? 'default' : 'pointer',
            }}
          >
            <div className="text-sm font-medium" style={{ color: 'var(--text-strong)' }}>{title}</div>
            <p className="mt-0.5 text-xs" style={{ color: 'var(--text-muted)' }}>{blurb}</p>
          </button>
        ))}
      </div>
      <p className="mt-2 text-xs" style={{ color: 'var(--text-faint)' }}>
        Either way, ask by voice: say <code>{agentName}</code> and then the question — for example{' '}
        <code>{agentName}, what did we decide about pricing?</code>
        {mode === 'chat' ? ' The answer appears in the meeting chat instead of being spoken.' : ''}
        {' '}Questions typed into the chat cannot be read during a call.
      </p>
      {error && <p className="mt-2 text-xs" style={{ color: 'var(--color-rose-700)' }}>{error}</p>}
    </div>
  )
}


function AgentForm({ form, update, onSubmit, saving, saved, error, submitLabel, promptHint, readOnly = false, showModality = true }) {
  return (
    // readOnly: a member who may look at the agent but not change it - the
    // fields stay visible, the inputs are disabled and there is no Save.
    <form onSubmit={onSubmit}>
      <fieldset disabled={readOnly} className="contents">
      <Field
        label="System prompt"
        hint={promptHint}
        htmlFor="sys-prompt"
      >
        <textarea
          id="sys-prompt"
          className="mc-input min-h-40 font-mono text-xs"
          value={form.system_prompt}
          onChange={(event) => update('system_prompt', event.target.value)}
        />
      </Field>

      <Field label="First message" hint="Posted into the meeting chat when the agent joins." htmlFor="first-msg">
        <textarea
          id="first-msg"
          className="mc-input min-h-20"
          value={form.first_message}
          onChange={(event) => update('first_message', event.target.value)}
        />
      </Field>

      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Provider" htmlFor="cfg-provider">
          <input
            id="cfg-provider"
            className="mc-input"
            value={form.provider}
            onChange={(event) => update('provider', event.target.value)}
          />
        </Field>
        <Field label="Model" htmlFor="cfg-model">
          <input
            id="cfg-model"
            className="mc-input"
            value={form.model}
            onChange={(event) => update('model', event.target.value)}
          />
        </Field>
        <Field label="Voice" htmlFor="cfg-voice">
          <input
            id="cfg-voice"
            className="mc-input"
            value={form.voice}
            onChange={(event) => update('voice', event.target.value)}
          />
        </Field>
        <Field label="Temperature" htmlFor="cfg-temp">
          <input
            id="cfg-temp"
            type="number"
            step="0.1"
            className="mc-input"
            value={form.temperature}
            onChange={(event) => update('temperature', event.target.value)}
          />
        </Field>
        {showModality && (
          <Field label="Response modality" htmlFor="cfg-modality" hint="How new agents made from this template answer.">
            <select
              id="cfg-modality"
              className="mc-input"
              value={form.response_modality}
              onChange={(event) => update('response_modality', event.target.value)}
            >
              <option value="">unchanged</option>
              {MODALITIES.map((value) => (
                <option key={value} value={value}>{value}</option>
              ))}
            </select>
          </Field>
        )}
      </div>

      <label className="mb-5 flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
        <input
          type="checkbox"
          checked={form.tool_results_to_chat}
          onChange={(event) => update('tool_results_to_chat', event.target.checked)}
        />
        Post tool results into the meeting chat
      </label>

      {error && (
        <div className="mb-4">
          <ErrorMessage title="Could not save" detail={error} />
        </div>
      )}

      </fieldset>
      {readOnly ? (
        <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
          You can view this but not change it. A workspace owner can grant that from the Members page.
        </p>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <button type="submit" className="mc-btn mc-btn-primary" disabled={saving}>
            {saving ? <Spinner size={14} /> : null} {submitLabel}
          </button>
          {saved && <span className="text-sm" style={{ color: 'var(--color-brand-600)' }}>Saved.</span>}
        </div>
      )}
    </form>
  )
}

export default function Agent() {
  const canManage = useCan('manage_agents')
  const isOwner = useIsOwner()
  const user = useUser()
  const [agents, setAgents] = useState(null)
  const [agentsError, setAgentsError] = useState(null)
  const [config, setConfig] = useState(null)
  const [configError, setConfigError] = useState(null)
  const [credentials, setCredentials] = useState(null)
  const [template, setTemplate] = useState(null)
  // 'template' edits the template agent; 'active' the live one; any other
  // value is an agent_config_id being viewed without switching to it.
  const [selected, setSelected] = useState('active')

  const [form, setForm] = useState(EMPTY_FORM)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [activatingId, setActivatingId] = useState(null)
  // What the last activation could not wire up (null when it all worked).
  const [wiringNotice, setWiringNotice] = useState(null)
  const [deletingId, setDeletingId] = useState(null)
  const [newOpen, setNewOpen] = useState(false)
  const [importOpen, setImportOpen] = useState(false)

  const loadAgents = useCallback(async () => {
    setAgentsError(null)
    try {
      const data = await listAgents()
      setAgents(data.agent_configs || [])
    } catch (err) {
      setAgents([])
      // A 400 means the member has no MeetStream API key yet — expected on a
      // new account, not an error worth showing (the empty-state UI already
      // guides them to create or import an agent).
      if (err.status !== 400) setAgentsError(err.message)
    }
  }, [])

  const loadConfig = useCallback(async (agentConfigId) => {
    setConfigError(null)
    setConfig(null)
    try {
      const data = await getAgent(agentConfigId)
      const cfg = data.agent_config || data
      const model = cfg.Model || {}
      const agent = cfg.Agent || {}
      setConfig(cfg)
      setForm({
        system_prompt: model.system_prompt || '',
        first_message: model.first_message || '',
        provider: model.provider || '',
        model: model.model || '',
        voice: model.voice || '',
        temperature: model.temperature ?? '',
        response_modality: agent.response_modality || '',
        tool_results_to_chat: Boolean(agent.tool_results_to_chat),
      })
    } catch (err) {
      setConfig(null)
      // A 404 for the default active agent means no agent exists yet — on a
      // new account this is the normal state, not a failure. Show the
      // template editor instead of a red error panel.
      if (!agentConfigId && err.status === 404) {
        setSelected('template')
        return
      }
      setConfigError(err.message)
    }
  }, [])

  const loadTemplate = useCallback(async () => {
    try {
      setTemplate(await getAgentTemplate())
    } catch {
      // The template has built-in defaults; the page still works without it.
    }
  }, [])

  useEffect(() => {
    loadAgents()
    loadConfig()
    loadTemplate()
    getAgentCredentials().then(setCredentials).catch(() => {})
  }, [loadAgents, loadConfig, loadTemplate])

  // Switching between the template and the active agent swaps what the
  // editor holds, so unsaved edits to one never leak into the other.
  useEffect(() => {
    if (selected !== 'template' || !template) return
    setForm({
      system_prompt: template.system_prompt || '',
      first_message: template.first_message || '',
      provider: template.provider || '',
      model: template.model || '',
      voice: template.voice || '',
      temperature: template.temperature ?? '',
      response_modality: template.response_modality || '',
      tool_results_to_chat: Boolean(template.tool_results_to_chat),
    })
    setSaved(false)
  }, [selected, template])

  function selectActive() {
    setSelected('active')
    setSaved(false)
    loadConfig()
  }

  function selectAgent(agent) {
    if (agent.IsActive) return selectActive()
    setSelected(agent.AgentConfigID)
    setSaved(false)
    loadConfig(agent.AgentConfigID)
  }

  const viewingId = selected === 'template' || selected === 'active' ? null : selected

  function update(key, value) {
    setForm((current) => ({ ...current, [key]: value }))
    setSaved(false)
  }

  async function save(event) {
    event.preventDefault()
    setSaving(true)
    setSaved(false)
    try {
      if (selected === 'template') {
        const next = await updateAgentTemplate({
          system_prompt: form.system_prompt,
          first_message: form.first_message,
          provider: form.provider || undefined,
          model: form.model || undefined,
          voice: form.voice || undefined,
          temperature: form.temperature === '' ? undefined : Number(form.temperature),
          response_modality: form.response_modality || undefined,
          tool_results_to_chat: form.tool_results_to_chat,
        })
        setTemplate(next)
        setSaved(true)
        return
      }
      await updateAgent({
        agent_config_id: viewingId || undefined,
        system_prompt: form.system_prompt,
        first_message: form.first_message,
        voice: form.voice || undefined,
        provider: form.provider || undefined,
        model: form.model || undefined,
        temperature: form.temperature === '' ? undefined : Number(form.temperature),
        response_modality: form.response_modality || undefined,
        tool_results_to_chat: form.tool_results_to_chat,
      })
      setSaved(true)
      await loadConfig(viewingId || undefined)
    } catch (err) {
      setConfigError(err.message)
    } finally {
      setSaving(false)
    }
  }

  async function activate(agentConfigId) {
    setActivatingId(agentConfigId)
    setWiringNotice(null)
    try {
      const result = await activateAgent(agentConfigId)
      // Activation used to report success while MeetStream had refused to
      // connect the agent to memory; say what it can actually do.
      const wiring = result?.wiring
      if (wiring?.problem) setWiringNotice(wiring.problem)
      else if (wiring && wiring.memory && !wiring.chat) {
        setWiringNotice('Connected to meeting memory, but MeetStream refused the "share in chat" tool, so the agent can answer out loud but not post into the chat.')
      }
      setSelected('active')
      await Promise.all([loadAgents(), loadConfig()])
    } catch (err) {
      setAgentsError(err.message)
    } finally {
      setActivatingId(null)
    }
  }

  async function remove(agent) {
    const name = agent.AgentName || 'this agent'
    if (!window.confirm(`Delete ${name}? It is removed from your MeetStream account too. Bots already in calls keep running; new ones cannot use it.`)) return
    setDeletingId(agent.AgentConfigID)
    setAgentsError(null)
    try {
      const result = await deleteAgent(agent.AgentConfigID)
      if (selected === agent.AgentConfigID || (result.was_active && selected === 'active')) setSelected('template')
      await Promise.all([loadAgents(), loadConfig()])
    } catch (err) {
      setAgentsError(err.message)
    } finally {
      setDeletingId(null)
    }
  }

  const refreshAll = () => Promise.all([loadAgents(), loadConfig()])

  return (
    <Page>
      <PageHeader
        title="Agent"
        description="The assistant MeetStream deploys into your calls — what it knows, how it sounds, and when it speaks."
        actions={
          canManage && (
            <>
              <button type="button" className="mc-btn mc-btn-secondary" onClick={() => setImportOpen(true)}>
                Import from MeetStream
              </button>
              <button type="button" className="mc-btn mc-btn-primary" onClick={() => setNewOpen(true)}>
                <PlusIcon size={16} /> New agent
              </button>
            </>
          )
        }
      />

      <div className="grid gap-5 lg:grid-cols-[20rem_1fr] items-start">
        <SectionCard title={`Your agents${agents ? ` (${agents.length})` : ''}`}>
          {/* The template is not a MeetStream agent - it lives in this app's
              config, cannot be deleted, and is what every new agent starts
              from. It is pinned above the real agents so it is always
              reachable, even before any exist. */}
          <button
            type="button"
            onClick={() => setSelected('template')}
            className="flex w-full items-center justify-between gap-3 px-5 py-3 text-left"
            style={{
              borderBottom: '1px solid var(--border-subtle)',
              backgroundColor: selected === 'template' ? 'var(--surface-sunken)' : 'transparent',
            }}
            aria-pressed={selected === 'template'}
          >
            <div className="flex min-w-0 items-center gap-3">
              <span className="mc-icon-chip" aria-hidden="true"><RobotIcon size={16} /></span>
              <div className="min-w-0">
                <div className="truncate text-sm font-medium" style={{ color: 'var(--text-strong)' }}>
                  Template agent
                </div>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  <Badge tone="brand">Template</Badge>
                  {template?.provider && <Badge>{template.provider}</Badge>}
                </div>
              </div>
            </div>
            <span className="text-xs" style={{ color: 'var(--text-faint)' }}>{isOwner ? 'Edit' : 'View'}</span>
          </button>

          {agentsError && (
            <div className="p-4">
              <ErrorMessage title="Could not load agents" detail={agentsError} onRetry={loadAgents} />
            </div>
          )}

          {agents === null ? (
            <Loading />
          ) : agents.length === 0 ? (
            <EmptyState
              icon={<AskAiIcon size={22} />}
              title="No agents yet"
              description="Create one, or import an agent that already exists on your MeetStream account."
              action={
                canManage && (
                  <button type="button" className="mc-btn mc-btn-secondary" onClick={() => setNewOpen(true)}>
                    Create an agent
                  </button>
                )
              }
            />
          ) : (
            <ul className="divide-y" style={{ borderColor: 'var(--border-subtle)' }}>
              {agents.map((agent) => (
                <li
                  key={agent.AgentConfigID}
                  className="flex items-center justify-between gap-3 px-5 py-3"
                  style={{
                    backgroundColor:
                      (agent.IsActive && selected === 'active') || selected === agent.AgentConfigID
                        ? 'var(--surface-sunken)'
                        : 'transparent',
                  }}
                >
                  <button
                    type="button"
                    className="min-w-0 flex-1 text-left"
                    onClick={() => selectAgent(agent)}
                    aria-pressed={selected === agent.AgentConfigID || (agent.IsActive && selected === 'active')}
                  >
                    <div className="truncate text-sm font-medium" style={{ color: 'var(--text-strong)' }}>
                      {agent.AgentName || 'Untitled agent'}
                    </div>
                    <div className="mt-1 flex flex-wrap gap-1.5">
                      {agent.Mode && <Badge>{agent.Mode}</Badge>}
                      {agent.Model?.provider && <Badge>{agent.Model.provider}</Badge>}
                    </div>
                  </button>
                  <div className="flex shrink-0 items-center gap-1.5">
                    {agent.IsActive ? (
                      <Badge tone="brand"><CheckIcon size={12} /> Active</Badge>
                    ) : canManage && (
                      <button
                        type="button"
                        className="mc-btn mc-btn-secondary"
                        onClick={() => activate(agent.AgentConfigID)}
                        disabled={activatingId === agent.AgentConfigID || deletingId === agent.AgentConfigID}
                      >
                        {activatingId === agent.AgentConfigID ? <Spinner size={13} /> : null} Use
                      </button>
                    )}
                    {canManage && (
                      <button
                        type="button"
                        className="mc-btn mc-btn-ghost px-2"
                        onClick={() => remove(agent)}
                        disabled={deletingId === agent.AgentConfigID || activatingId === agent.AgentConfigID}
                        title="Delete this agent"
                        aria-label={`Delete ${agent.AgentName || 'agent'}`}
                      >
                        {deletingId === agent.AgentConfigID ? <Spinner size={13} /> : <TrashIcon size={14} />}
                      </button>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </SectionCard>

        <div className="min-w-0">
          {selected === 'template' ? (
            <Card>
              <div className="mb-1 flex flex-wrap items-center gap-2">
                <h2 className="text-base font-semibold">Template agent</h2>
                <Badge tone="brand">Template</Badge>
              </div>
              <p className="mb-5 text-sm" style={{ color: 'var(--text-muted)' }}>
                Every new agent starts as a copy of this. It cannot be deleted, but everything about it — provider,
                model, voice, prompt and first message — can be changed. Use <code>{'{agent_name}'}</code> where the
                agent's name should appear.
              </p>
              <AgentForm
                form={form}
                update={update}
                onSubmit={save}
                saving={saving}
                saved={saved}
                error={configError}
                submitLabel="Save template"
                promptHint="The starting system prompt for new agents."
                readOnly={!isOwner}
              />
            </Card>
          ) : configError && !config ? (
            <Card>
              <ErrorMessage title="No agent configured" detail={configError} onRetry={loadConfig} />
              <p className="mt-4 text-sm" style={{ color: 'var(--text-muted)' }}>
                Create an agent or activate one from the list to configure it here.
              </p>
            </Card>
          ) : !config ? (
            <Card><Loading /></Card>
          ) : (
            <Card>
              <div className="mb-5 flex flex-wrap items-center gap-2">
                <h2 className="text-base font-semibold">{config.AgentName || 'Agent'}</h2>
                {viewingId ? <Badge>Not active</Badge> : <Badge tone="brand"><CheckIcon size={12} /> Active</Badge>}
                {config.Mode && <Badge>{config.Mode}</Badge>}
                {config.AgentConfigID && (
                  <span className="font-mono text-[0.7rem]" style={{ color: 'var(--text-faint)' }}>
                    {config.AgentConfigID}
                  </span>
                )}
                {viewingId && canManage && (
                  <button
                    type="button"
                    className="mc-btn mc-btn-secondary ml-auto"
                    onClick={() => activate(viewingId)}
                    disabled={activatingId === viewingId}
                  >
                    {activatingId === viewingId ? <Spinner size={13} /> : null} Use this agent
                  </button>
                )}
              </div>

              {(wiringNotice || config.MemoryProblem) && (
                <div className="mb-4">
                  <Notice title="This agent can't look anything up in a call yet">
                    {wiringNotice || config.MemoryProblem}
                    <br />
                    <Link to="/settings?section=meetings" className="mt-1 inline-block font-semibold underline">Set the public address in Settings → Meetings</Link>
                  </Notice>
                </div>
              )}

              <InteractionModePicker
                agentConfigId={config.AgentConfigID}
                agentName={config.AgentName || 'Meet Companion'}
                mode={config.InteractionMode || 'voice'}
                readOnly={!canManage}
                onChanged={() => {
                  // The modality changed on MeetStream: reload so the form
                  // and the list both show what is now saved there.
                  loadAgents()
                  loadConfig(viewingId || undefined)
                }}
              />

              <AgentForm
                form={form}
                update={update}
                onSubmit={save}
                saving={saving}
                saved={saved}
                error={configError}
                submitLabel="Save agent"
                promptHint="Controls when the agent speaks and how it answers."
                readOnly={!canManage}
                showModality={false}
              />
            </Card>
          )}

          {credentials && (
            <Card className="mt-5">
              <h2 className="mb-1 text-base font-semibold">Connection</h2>
              <p className="mb-4 text-sm" style={{ color: 'var(--text-muted)' }}>
                What this workspace's agent is wired to. Secrets are shown masked.
              </p>
              <dl className="divide-y text-sm" style={{ borderColor: 'var(--border-subtle)' }}>
                {Object.entries(credentials).map(([key, value]) => {
                  let display = '—'
                  if (typeof value === 'object' && value !== null) {
                    if (key === 'memory_extraction_llm') {
                      if (value.masked_value) {
                        display = value.masked_value
                      } else if (value.provider) {
                        display = value.model ? `${value.provider} (${value.model})` : value.provider
                      } else if (user?.workspace_ai?.mode === 'workspace' && user.workspace_ai.provider) {
                        display = user.workspace_ai.model
                          ? `${user.workspace_ai.provider} (${user.workspace_ai.model})`
                          : user.workspace_ai.provider
                      } else if (value.configured || value.api_key_configured) {
                        display = 'configured'
                      } else {
                        display = 'not set'
                      }
                    } else {
                      display = value.masked_value || (value.configured ? 'configured' : 'not set')
                    }
                  } else {
                    display = String(value ?? '—')
                  }

                  return (
                    <div key={key} className="flex flex-wrap justify-between gap-3 py-2.5">
                      <dt style={{ color: 'var(--text-muted)' }}>{key.replace(/_/g, ' ')}</dt>
                      <dd className="font-mono text-xs" style={{ color: 'var(--text-strong)' }}>
                        {display}
                      </dd>
                    </div>
                  )
                })}
              </dl>
            </Card>
          )}
        </div>
      </div>

      <NewAgentModal open={newOpen} onClose={() => setNewOpen(false)} onCreated={refreshAll} template={template} />
      <ImportAgentsModal open={importOpen} onClose={() => setImportOpen(false)} onImported={refreshAll} />
    </Page>
  )
}
