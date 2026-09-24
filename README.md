<div align="center">

<img src="assets/branding/logo-icon.png" alt="Meet Companion logo" width="112" />

# Meet Companion

**A bot joins your call. What was said comes back as decisions, action items and searchable notes — on your machine, with the AI model and database you choose.**

[![Latest release](https://img.shields.io/github/v/release/ThalhaAhamed/MeetCompanion?label=release&color=3b4873)](https://github.com/ThalhaAhamed/MeetCompanion/releases/latest)
[![CI](https://img.shields.io/github/actions/workflow/status/ThalhaAhamed/MeetCompanion/ci.yml?label=CI)](https://github.com/ThalhaAhamed/MeetCompanion/actions)
![Platforms](https://img.shields.io/badge/Windows%20%7C%20macOS%20%7C%20Linux-3b4873)
[![License: MIT](https://img.shields.io/badge/license-MIT-e3b1bc)](LICENSE)

[Quick start](#-quick-start) · [Demo](#-demo) · [Architecture](#%EF%B8%8F-architecture) · [Docker](#-docker) · [Configuration](#%EF%B8%8F-configuration) · [Roadmap](#%EF%B8%8F-roadmap) · [FAQ](#-faq)

</div>

<div align="center">
<img src="docs/media/meeting-to-note.gif" alt="A transcript is pasted, processed, and becomes a summary, action items, memories and a note" width="900" />
</div>

Send a bot into your call via **[MeetStream](https://meetstream.ai)** (Google Meet, Zoom, Teams) — as soon as the meeting finishes, Meet Companion **automatically retrieves the transcript**, runs it through your chosen LLM to extract summaries, key decisions, and action items with deadlines, and files everything into organized Markdown notes. You can also paste or import existing transcripts directly without needing an account. Everything after the call runs on infrastructure you own: meetings become **Markdown notes with live action-item checkboxes**, searchable **semantically**, and answerable with **Ask AI** — plus an in-call agent that can recall "what did we decide last time?" *during* your next meeting. Built for teams and individuals who run their own tools and want meeting intelligence that stays completely theirs.

## 🚀 Quick start

**[⬇ Download the latest release](https://github.com/ThalhaAhamed/MeetCompanion/releases/latest)** for Windows, macOS or Linux — open it, follow the five-step setup, done. Local SQLite and a local Ollama model need no keys.

```bash
docker compose up -d      # self-hosted instead → http://localhost:8000
```

## 📥 Installation

| Platform | Get it | First launch |
|---|---|---|
| **Windows** | `MeetCompanion-Windows-Setup.exe` | SmartScreen: **More info → Run anyway** (unsigned build) |
| **macOS** | `…-macOS-arm64.dmg` (Apple silicon) · `…-x64.dmg` (Intel) | Drag to Applications, **right-click → Open** once |
| **Linux** | `…-Linux-x86_64.AppImage` | `chmod +x`, run |
| **Docker** | `docker compose up -d` | <http://localhost:8000> |
| **Source** | Python 3.12 + Node 22 | `python scripts/dev.py` — [docs/development.md](docs/development.md) |

> [!TIP]
> No `.env`, no database setup, no migrations: first launch configures everything and the app creates and upgrades its own schema. Data locations per OS: [docs/installation.md](docs/installation.md).

## 💡 Why Meet Companion?

- **Bring your own AI** — OpenAI, Anthropic, Gemini, Groq, xAI, a local [Ollama](https://ollama.com) model, or any OpenAI-compatible endpoint. Switch any time.
- **Your data stays where you put it** — SQLite by default; PostgreSQL, Supabase, Neon or Railway when you want it. Embeddings are computed locally.
- **Notes, not a feed** — each meeting is filed as a Markdown note under `Meetings / year / month` with live checkboxes.
- **Fully offline is real** — Ollama + SQLite means nothing leaves your computer and no API key is needed.

## ✨ Features

- 🎥 **Meeting capture** — automatic bot joining for Google Meet, Zoom, or Teams with instant transcript retrieval and processing; or manually paste/import transcripts anytime.
- 🧠 **Memory extraction** — decisions, commitments, requirements, concerns, open questions, action items with owners and due dates.
- 📓 **Notebook** — folders, tags, favourites, search; rendered Markdown with an editor behind it.
- ✅ **Action items in sync** — tick a box in a note and the task completes; `- [ ] call Bob` in any note becomes a real task.
- 🔍 **Semantic search** — hybrid vector + keyword search across everything ever said.
- 💬 **Ask AI** — answers grounded in notes, meetings and uploaded documents (PDF, Word, Markdown, CSV), with sources; never invents.
- 🕸️ **Knowledge graph** — meetings, people, decisions and action items as an explorable graph.
- 🎙️ **In-call recall** — a built-in [MCP](https://modelcontextprotocol.io) server the in-meeting agent queries live.
- 👥 **Workspaces** — share by join code, owners approve who gets in and decide what members may add, edit, delete, export or invite.

## 📸 Screenshots

<div align="center">
<img src="docs/screenshots/dashboard.png" alt="Dashboard: open action items, today's meetings, recent notes" width="900" />
<img src="docs/screenshots/settings-database.png" alt="Database settings: current database shown as Connected, provider picker" width="900" />
<img src="docs/screenshots/agent.png" alt="Agent configuration: template prompt, provider, model, voice" width="900" />
</div>

## 🎬 Demo

**Ask AI** — grounded answer with an owner table and source links:

<div align="center"><img src="docs/media/ask-ai.gif" alt="Asking a question and getting an answer with sources" width="900" /></div>

**Knowledge graph** — select a meeting, then filter by type:

<div align="center"><img src="docs/media/knowledge-graph.gif" alt="Selecting a graph node and toggling type filters" width="900" /></div>

**Workspaces** — owner permissions, create a second workspace, switch:

<div align="center"><img src="docs/media/workspaces.gif" alt="Toggling member permissions and switching workspaces" width="900" /></div>

First-run wizard: [docs/installation.md](docs/installation.md).

## 🏗️ Architecture

```
  Desktop (Electron)  ──┐
  Browser             ──┼──▶  React UI  ──▶  FastAPI  ──▶  Services  ──▶  Providers
  MeetStream agent  ──▶  MCP server  ─┘        (auth gate, permissions)      ├─ LLM: OpenAI · Anthropic · Gemini · Groq · xAI · Ollama · compatible
  MeetStream webhooks ─▶ /api/webhooks (HMAC)                                 ├─ Database: SQLite (numpy search) · PostgreSQL (pgvector + tsvector)
                                                                               └─ MeetStream API (bots, transcripts)
```

**Request flow.** Every request passes an auth gate (signed session cookie, or the desktop device key) and, for anything that changes data, a per-workspace permission check. Every query is scoped to the caller's workspace (`organization_id`) in the repository layer — there is no path that reads a row by id alone.

**Data flow for a meeting.** Transcript (webhook or upload) → segments stored → participants → LLM extraction into memories and action items (structured JSON, validated; rule-based fallback if the provider fails) → chunked and embedded locally with `all-MiniLM-L6-v2` (384-d) → a Markdown note is written and filed → everything is searchable (reciprocal-rank fusion of vector and keyword hits) and answerable by Ask AI and by the MCP tools.

**Providers are interfaces, not conditionals.** A new LLM vendor is one adapter in `app/providers/llm/` plus a registry entry; the adapter's declared fields *are* the settings form. **The database is swapped, not abstracted away** — only similarity and keyword search differ, and those live in `app/providers/database/`, chosen from the live connection's dialect. Schema is Alembic-managed and upgraded at startup. Full detail: [docs/architecture.md](docs/architecture.md).

## 🗂️ Project structure

```
app/            FastAPI backend — api/ (one router per page) · services/ (pipeline) · providers/ (llm/, database/)
                database/ (org-scoped repositories, bootstrap) · models/ · migrations/ (Alembic) · mcp/ · rag/
frontend/src/   React app — pages/ (one per route) · components/ · api.js (all HTTP calls)
desktop/        Electron shell + PyInstaller spec
tests/          pytest (hermetic SQLite) · frontend/src/__tests__ (vitest)
docs/           documentation, screenshots, demo media
```

## 🐳 Docker

```bash
docker compose up -d                      # app on http://localhost:8000, SQLite in a volume
docker compose --profile postgres up -d   # plus a pgvector Postgres the app is pointed at
```

- **First visit** walks through onboarding; configuration, database and embedding model live in the `data` volume.
- **Skip onboarding** by setting `LLM_PROVIDER`, `LLM_API_KEY`, `DATABASE_URL` in `docker-compose.yml`.
- **Production mode** in the image (`/docs` off); the image is built and booted in CI on every push.

> [!IMPORTANT]
> To send bots into calls, MeetStream must reach your server — a public host or a tunnel ([docs/live-meetings.md](docs/live-meetings.md)). Upload and Ask AI work without it.

## ⚙️ Configuration

Set from **Settings** in the app, saved to `data/config.json`. An **environment variable that is set always wins** (the UI shows those fields read-only), so containers stay reproducible.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY` | from Settings | The AI for extraction and Ask AI |
| `DATABASE_URL` | local SQLite | `postgresql://…` for a shared or hosted database |
| `MEETSTREAM_API_KEY` | per member | Lets the app send bots into calls |
| `MCP_SERVER_URL` | — | Public URL MeetStream reaches your server on |
| `ALLOW_SELF_SIGNUP` | `true` | `false` on a public server: only owners add members |

Every variable: [docs/configuration.md](docs/configuration.md).

## 🧰 Technology stack

**Backend** Python 3.12 · FastAPI · SQLAlchemy 2 (async) · Alembic · SQLite / PostgreSQL + pgvector · **AI** adapters over `httpx`, local ONNX embeddings · **Frontend** React 19 · Vite · Tailwind CSS 4 · **Desktop** Electron + PyInstaller · **Integration** MeetStream bots, HMAC-signed webhooks, MCP server. Why each: [docs/development.md#tech-stack](docs/development.md#tech-stack).

## 📦 Dependencies

Pinned in [`requirements.txt`](requirements.txt), [`frontend/package.json`](frontend/package.json) and [`desktop/package.json`](desktop/package.json). No vendor SDKs; `pip-audit` runs in CI. The 90 MB embedding model downloads on first use; Postgres drivers are only exercised when you use Postgres.

## 🧪 Running tests

```bash
.venv/Scripts/python -m pytest      # 246 backend tests, throwaway SQLite, no network (~3 min)
npm --prefix frontend test          # 15 vitest tests
npm --prefix frontend run lint      # oxlint, 0 errors
```

## 🔁 CI/CD

- **Every push / PR** ([`ci.yml`](.github/workflows/ci.yml)): backend suite on **SQLite and PostgreSQL + pgvector**, frontend lint/tests/build, `pip-audit`, Docker image built and booted.
- **Every `v*` tag** ([`release.yml`](.github/workflows/release.yml)): version stamped from the tag, server frozen with PyInstaller, Windows/macOS/Linux installers built and attached to a GitHub release — ~40 min from tag to published.

## 🗺️ Roadmap

- [ ] Code-signed Windows and macOS builds
- [ ] MySQL / MariaDB support
- [ ] Re-indexing after changing embedding models
- [ ] Desktop auto-update

## 🩺 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Meeting stuck at *Extracting…* | MeetStream cannot reach your server | Public URL or tunnel in `MCP_SERVER_URL`; the next launch re-points the agent |
| *Processed without AI (…)* | Provider failed; the bracket says why | Fix it in Settings (e.g. `ollama pull <model>`), then **Reprocess** |
| Database *Not reachable* | Host, credentials or SSL | The banner carries the driver's message; Neon/Supabase strings paste as-is |
| Blank page after upgrading from v0.2 | Accounts predating workspaces | Fixed in v0.3.1 — upgrade |

More: [docs/troubleshooting.md](docs/troubleshooting.md).

## ❓ FAQ

**Does my data leave my machine?** Only where you point it: SQLite + Ollama, nothing leaves; embeddings are always local — [docs/security.md](docs/security.md).
**Do I need MeetStream?** Only to send a bot into a call. Upload, search and Ask AI work without it.
**Which AI models?** OpenAI, Anthropic, Gemini, Groq, xAI, Ollama, any OpenAI-compatible endpoint. *Test connection* checks the key **and** the model.
**Can my team share one workspace?** Yes — one shared Postgres, a join code, owner-set permissions — [docs/workspaces.md](docs/workspaces.md).
**Is it free?** MIT, no accounts, no telemetry. You pay only the AI or database provider you choose, or nothing with Ollama + SQLite.

## 📚 Documentation

[Installation](docs/installation.md) · [Configuration](docs/configuration.md) · [Live meetings](docs/live-meetings.md) · [Workspaces & export](docs/workspaces.md) · [Security & privacy](docs/security.md) · [Troubleshooting](docs/troubleshooting.md) · [For developers](docs/development.md) · [Architecture](docs/architecture.md) · [Agent guide](AGENTS.md) · [Security policy](SECURITY.md)

## 🤝 Contributing

Issues, pull requests, providers and docs are welcome — start with [CONTRIBUTING.md](CONTRIBUTING.md). A new LLM provider is one file in `app/providers/llm/`. Run the tests before opening a pull request; commits follow [Conventional Commits](https://www.conventionalcommits.org/).

## 📄 License

[MIT](LICENSE). Built on [MeetStream](https://meetstream.ai), [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via [fastembed](https://github.com/qdrant/fastembed), FastAPI, SQLAlchemy, React, Vite, Tailwind and Electron.
