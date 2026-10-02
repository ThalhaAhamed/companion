# Working in this repository

Guidance for AI coding agents (and humans in a hurry). `docs/architecture.md`
explains *why* things are shaped this way; this file is the *how*.

## What it is

Meet Companion sends a [MeetStream](https://meetstream.ai) bot into a call,
extracts memories and action items from the transcript with a pluggable LLM,
files the result as Markdown notes, and answers questions over everything -
including from inside the next call, through a built-in MCP server. It runs
as a self-hosted web app and as an Electron desktop app; same code.

## Map

```
app/api/           FastAPI routers, one per UI page   (meetings.py, notebook.py, members.py, ...)
app/services/      application logic                  (processing.py = the post-call pipeline)
app/providers/     swappable adapters                  (llm/<vendor>.py, database/<dialect>.py)
app/database/      repositories (org-scoped) + bootstrap.py (startup schema/data repairs)
app/models/        SQLAlchemy models (database.py), Pydantic schemas (schemas.py), portable column types
app/migrations/    Alembic revisions - the schema's source of truth
app/mcp/           the MCP server the in-call agent talks to (tools.py)
app/middleware/    auth gate, body/rate limits
app/permissions.py what workspace members may do; owners always may
app/rag/           chunking, embedding index, hybrid retrieval
frontend/src/      React 19 + Vite + Tailwind: pages/ (one per route), components/, api.js (all HTTP calls)
desktop/           Electron shell around the PyInstaller-frozen server
tests/             pytest, hermetic (throwaway SQLite, no network); tests/conftest.py has the fixtures
scripts/           dev.py (run both servers), reset_password.py, importers, release stamping
```

## Run, test, verify

```bash
py -3.12 -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # once
npm --prefix frontend install                                                        # once
.venv/Scripts/python scripts/dev.py          # API :8000 (reload) + UI :3000; no .env needed

.venv/Scripts/python -m pytest -q            # ~3 min, 230+ tests, no Postgres/Docker/network needed
npm --prefix frontend run lint               # oxlint - must be 0 errors
npm --prefix frontend test                   # vitest
npm --prefix frontend run build              # must be warning-free
```

CI (`.github/workflows/ci.yml`) runs pytest on SQLite **and** Postgres, the
frontend checks, `pip-audit`, and builds + boots the Docker image. A change
is done when all of that is green, not when the code compiles.

The backend started by `scripts/dev.py` reloads on save; a backend started
any other way must be restarted to pick up Python changes.

## Rules that are easy to miss

1. **Every route that changes data takes a permission.**
   `@router.post(..., dependencies=[Depends(perms.require("edit_content"))])`
   with one of: `create_content`, `edit_content`, `delete_content`,
   `manage_agents`, `export_workspace`, `invite_members` (`app/permissions.py`).
   Owners always pass. In the UI, hide the control with `useCan('...')`
   from `frontend/src/user.js`. Server-wide configuration is owner-only via
   `require_owner` / `require_setup_access`.
2. **Every query is workspace-scoped.** Take `org_id: uuid.UUID =
   Depends(get_current_org_id)` and pass it to the repository. Never look a
   row up by id alone.
3. **A model change is a migration.** Edit `app/models/database.py`, then
   `alembic revision --autogenerate -m "..."` and review the file under
   `app/migrations/versions/`. `tests/test_migrations.py` fails on drift. Do
   not add schema patches to `bootstrap.py`.
4. **A new MCP tool touches four places in `app/mcp/tools.py`:** its schema
   in `MCP_TOOL_DEFINITIONS`, its branch in `execute_tool`, its rendering in
   `format_tool_output_text`, and `WRITE_TOOLS` if it changes data (write
   tools are on by default; an owner can switch them off in Settings →
   Meetings). Arguments are validated against the tool's `inputSchema`
   before dispatch. Add a case to `tests/test_mcp.py`.
5. **A new LLM provider is one file** in `app/providers/llm/` subclassing
   `LLMProvider` (or `OpenAICompatibleProvider` with a different base URL),
   plus a `ProviderDescriptor` in `app/providers/llm/__init__.py` listing
   only the fields it needs - that metadata *is* the settings form. Test in
   `tests/test_llm_providers.py` with `pytest_httpx`.
6. **Secrets never come back out.** Responses carry `mask_secret(...)`
   previews (`app/runtime_config.py`). An omitted key on save keeps the
   stored one. Never log a key, token or password.
7. **Timestamps leave the API with an explicit UTC offset.** Response
   schemas inherit `SchemaBase`; hand-built dicts use `utc_iso(...)` from
   `app/models/schemas.py`. SQLite returns naive datetimes, and a naive ISO
   string is read by browsers as local time.
8. **Errors must say what to do.** A provider or database failure is
   turned into a sentence a person can act on (see `_friendly_db_error`,
   `transcript_unavailable_message`), never a raw exception string.
9. **Frontend calls go through `frontend/src/api.js`.** Add a function
   there (JSON body via `JSON.stringify`, same as its neighbours); pages
   never call `fetch` directly. UI primitives live in
   `frontend/src/components/ui.jsx` (`Card`, `Badge`, `Field`, `Modal`,
   `ErrorMessage`, `EmptyState`, `Spinner`, ...); use them rather than
   restyling.

## Tests

- `authed_client` (tests/conftest.py) is a client with a real signed session
  for a fresh owner; `client` is anonymous. `tests/test_security.py` has
  `_signup(client, email, workspace=..., join_code=...)` for multi-user
  scenarios.
- The schema is rebuilt per test; rows never leak between tests.
- Mock providers with `pytest_httpx` (`httpx_mock`), never by monkeypatching
  `httpx` itself. MeetStream calls are mocked by monkeypatching
  `meetstream_client.<method>`.
- A regression test states, in its docstring, what was observed and why it
  mattered. That is the convention here.

## Things to know before touching them

- `app/services/processing.py:process_meeting_transcript` is the whole
  post-call pipeline (fetch → store → extract → embed → note → finalise).
  Failures short of a crash must still leave the meeting in a truthful state.
- `app/api/webhooks.py` accepts MeetStream deliveries. With a signing secret
  configured they are HMAC-verified; without one, only events for bots this
  install launched are accepted. Events are idempotent on
  `(bot_id, event_type, timestamp)` (a hash of the body when there is no
  timestamp), and a lifecycle event only ever moves a meeting forward.
- Reprocess replaces a meeting's memories in the same transaction that
  stores the new ones, and matches its extracted action items against the
  new run (`ActionItemRepository.replace_extracted`): ticks and tasks written
  by hand survive. A note a person edited (`Note.edited_by_user`) keeps its
  text; only its task lines are brought up to date.
- `app/middleware/auth_gate.py` decides what is reachable signed-out
  (`EXEMPT_PREFIXES`, `BOOT_PATHS`) and when first-run setup is open (only
  with no saved config **and** no account). Changing this is a security
  change; `tests/test_security.py` and `tests/test_setup.py` cover it.
- The desktop shell signs in with a per-machine device key
  (`mc_device` cookie) that maps to the sole owner - see
  `tests/test_device_login.py` for the rules.

## Style

Match the surrounding file. Comments explain *why* - usually the bug that
motivated the code - not what the next line does. Commit messages are
Conventional Commits (`feat:`, `fix:`, `docs:`, `ci:`) and their bodies say
what was observed, not just what changed. No AI attribution lines.
