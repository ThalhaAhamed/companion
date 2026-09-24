/**
 * Shared interface primitives.
 *
 * Small, unopinionated building blocks so pages stay readable and every screen
 * handles loading, empty and error states the same way.
 */
import { useEffect, useRef, useState } from 'react'
import { AlertIcon, CloseIcon, DownloadIcon, InboxIcon } from './Icons'

export function Card({ children, className = '', padded = true, ...rest }) {
  return (
    <div className={`mc-panel ${padded ? 'p-5' : ''} ${className}`} {...rest}>
      {children}
    </div>
  )
}

export function SectionCard({ title, action, children, className = '' }) {
  return (
    <Card padded={false} className={className}>
      <div className="flex items-center justify-between gap-3 px-5 py-4">
        <h2 className="text-[0.95rem] font-semibold">{title}</h2>
        {action}
      </div>
      <div style={{ borderTop: '1px solid var(--border-subtle)' }}>{children}</div>
    </Card>
  )
}

export function Spinner({ size = 18 }) {
  return (
    <span
      className="inline-block animate-spin rounded-full align-[-2px]"
      style={{
        width: size,
        height: size,
        border: '2px solid var(--border-default)',
        borderTopColor: 'var(--brand-solid)',
      }}
      role="status"
      aria-label="Loading"
    />
  )
}

export function Loading({ label = 'Loading…' }) {
  return (
    <div className="flex items-center gap-3 py-10 justify-center" style={{ color: 'var(--text-muted)' }}>
      <Spinner />
      <span className="text-sm">{label}</span>
    </div>
  )
}

export function ErrorMessage({ title = 'Something went wrong', detail, onRetry }) {
  return (
    <div
      className="flex items-start gap-3 rounded-xl p-4 text-sm"
      style={{
        backgroundColor: 'var(--danger-bg)',
        border: '1px solid var(--danger-border)',
        color: 'var(--danger-fg)',
      }}
      role="alert"
    >
      <AlertIcon size={18} />
      <div className="min-w-0 flex-1">
        <div className="font-semibold">{title}</div>
        {detail && <div className="mt-0.5 break-words opacity-90">{detail}</div>}
      </div>
      {onRetry && (
        <button type="button" className="mc-btn mc-btn-secondary" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  )
}

export function EmptyState({ icon, title, description, action }) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-14 text-center">
      <div
        className="mb-4 flex h-12 w-12 items-center justify-center rounded-full"
        style={{ backgroundColor: 'var(--surface-raised)', color: 'var(--text-faint)' }}
      >
        {icon || <InboxIcon size={22} />}
      </div>
      <h3 className="text-base font-semibold">{title}</h3>
      {description && (
        <p className="mt-1 max-w-sm text-sm" style={{ color: 'var(--text-muted)' }}>
          {description}
        </p>
      )}
      {action && <div className="mt-5">{action}</div>}
    </div>
  )
}

export function Badge({ children, tone = 'neutral' }) {
  const tones = {
    neutral: {},
    brand: { backgroundColor: 'var(--brand-soft)', color: 'var(--brand-soft-text)', borderColor: 'transparent' },
    success: { backgroundColor: 'var(--color-lavender-200)', color: 'var(--color-brand-800)', borderColor: 'transparent' },
    warning: { backgroundColor: 'var(--color-peach-200)', color: 'var(--color-peach-700)', borderColor: 'transparent' },
    danger: { backgroundColor: 'var(--color-rose-200)', color: 'var(--color-rose-700)', borderColor: 'transparent' },
  }
  return (
    <span className="mc-badge" style={tones[tone] || {}}>
      {children}
    </span>
  )
}

/**
 * Something the person should know before it bites them - not an error, the
 * action still works. Peach, the design system's warning colour.
 */
export function Notice({ title, children }) {
  return (
    <div
      role="status"
      className="rounded-lg px-3 py-2 text-xs"
      style={{ backgroundColor: 'var(--color-peach-200)', color: 'var(--color-peach-700)' }}
    >
      {title && <strong className="block">{title}</strong>}
      {children}
    </div>
  )
}

/** Live state of a connection, as a dot + word: Connected / Not reachable / Checking… */
export function ConnectionBadge({ connected }) {
  const state =
    connected === true
      ? { label: 'Connected', dot: 'var(--color-brand-800)', tone: 'success' }
      : connected === false
        ? { label: 'Not reachable', dot: 'var(--color-rose-700)', tone: 'danger' }
        : { label: 'Checking…', dot: 'var(--text-faint)', tone: 'neutral' }
  return (
    <Badge tone={state.tone}>
      <span
        aria-hidden="true"
        className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle"
        style={{ backgroundColor: state.dot }}
      />
      {state.label}
    </Badge>
  )
}

export function Field({ label, hint, error, children, htmlFor }) {
  return (
    <div className="mb-4">
      {label && (
        <label className="mc-label" htmlFor={htmlFor}>
          {label}
        </label>
      )}
      {children}
      {hint && !error && (
        <p className="mt-1 text-xs" style={{ color: 'var(--text-faint)' }}>
          {hint}
        </p>
      )}
      {error && (
        <p className="mt-1 text-xs" style={{ color: 'var(--color-rose-700)' }}>
          {error}
        </p>
      )}
    </div>
  )
}

export function Modal({ open, onClose, title, children, footer, width = '32rem' }) {
  useEffect(() => {
    if (!open) return undefined
    function onKey(event) {
      if (event.key === 'Escape') onClose?.()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ backgroundColor: 'rgba(18, 21, 36, 0.55)' }}
      onMouseDown={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={typeof title === 'string' ? title : undefined}
    >
      <div
        className="mc-panel w-full max-h-[90vh] overflow-y-auto mc-scroll"
        style={{ maxWidth: width, backgroundColor: 'var(--surface-panel)' }}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div
          className="flex items-center justify-between px-5 py-4"
          style={{ borderBottom: '1px solid var(--border-subtle)' }}
        >
          <h2 className="text-base font-semibold">{title}</h2>
          <button type="button" className="mc-btn mc-btn-ghost px-2" onClick={onClose} aria-label="Close">
            <CloseIcon size={18} />
          </button>
        </div>
        <div className="p-5">{children}</div>
        {footer && (
          <div
            className="flex justify-end gap-2 px-5 py-4"
            style={{ borderTop: '1px solid var(--border-subtle)' }}
          >
            {footer}
          </div>
        )}
      </div>
    </div>
  )
}

// Theme-aware so glyphs stay readable in both light and dark; see the
// --chip-* tokens in index.css.
const TONES = {
  brand: { fg: 'var(--chip-brand-fg)', bg: 'var(--chip-brand-bg)' },
  lavender: { fg: 'var(--chip-lavender-fg)', bg: 'var(--chip-lavender-bg)' },
  peach: { fg: 'var(--chip-peach-fg)', bg: 'var(--chip-peach-bg)' },
  rose: { fg: 'var(--chip-rose-fg)', bg: 'var(--chip-rose-bg)' },
}

export function IconChip({ icon, tone = 'brand', size = 40 }) {
  const palette = TONES[tone] || TONES.brand
  return (
    <span
      className="mc-icon-chip"
      style={{ backgroundColor: palette.bg, color: palette.fg, width: size, height: size }}
      aria-hidden="true"
    >
      {icon}
    </span>
  )
}

export function StatTile({ label, value, hint, icon, tone = 'brand' }) {
  return (
    <Card className="mc-panel-interactive">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[0.78rem] font-medium" style={{ color: 'var(--text-muted)' }}>
            {label}
          </div>
          <div
            className="mt-2 text-[1.9rem] font-semibold leading-none tracking-tight"
            style={{ color: 'var(--text-strong)' }}
          >
            {value}
          </div>
          {hint && (
            <div className="mt-2 text-xs" style={{ color: 'var(--text-faint)' }}>
              {hint}
            </div>
          )}
        </div>
        {icon && <IconChip icon={icon} tone={tone} />}
      </div>
    </Card>
  )
}


/**
 * Download-as menu.
 *
 * The options are plain anchors, not buttons: the browser then handles the
 * download itself (cookie sent, server filename honoured) instead of the app
 * pulling the file into memory to re-save it.
 */
export function ExportMenu({ options, label = 'Export', compact = false, align = 'right' }) {
  const [open, setOpen] = useState(false)
  const box = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    const close = (event) => {
      if (!box.current?.contains(event.target)) setOpen(false)
    }
    const onKey = (event) => event.key === 'Escape' && setOpen(false)
    document.addEventListener('mousedown', close)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', close)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div className="relative" ref={box}>
      <button
        type="button"
        className={compact ? 'mc-btn mc-btn-ghost px-2' : 'mc-btn mc-btn-secondary'}
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={label}
        title={label}
      >
        <DownloadIcon size={compact ? 17 : 15} />
        {compact ? null : <span className="ml-1.5">{label}</span>}
      </button>
      {open && (
        <div
          role="menu"
          className="mc-panel absolute z-20 mt-1 min-w-[12rem] overflow-hidden p-1"
          style={{ [align]: 0, boxShadow: 'var(--shadow-lg, 0 10px 30px rgba(0,0,0,.25))' }}
        >
          {options.map((option) => (
            <a
              key={option.href}
              role="menuitem"
              href={option.href}
              download
              className="mc-nav-item block w-full px-3 py-2 text-left text-sm"
              onClick={() => setOpen(false)}
            >
              {option.label}
            </a>
          ))}
        </div>
      )}
    </div>
  )
}
