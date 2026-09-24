# Configuration

## Settings and environment variables

Everything is configurable from **Settings** in the app. Precedence is:

1. process environment variables (containers, CI) — shown read-only in Settings
2. what you saved in the UI
3. `.env` file (self-hosted convenience)
4. built-in defaults

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite+aiosqlite:///data/meet-companion.db` | Storage; `postgresql+asyncpg://…` for Postgres |
| `LLM_PROVIDER` | `ollama` | `openai` · `anthropic` · `gemini` · `ollama` · `groq` · `openai_compatible` |
| `LLM_MODEL` | per provider | Model name |
| `LLM_API_KEY` | — | Hosted providers only |
| `LLM_BASE_URL` | per provider | Proxies, gateways, self-hosted endpoints |
| `MEETSTREAM_API_KEY` | — | Deployment-wide key; each member can also add their own in Settings |
| `MEETSTREAM_WEBHOOK_SECRET` | — | Signature check for webhook deliveries (also settable in Settings) |
| `MCP_SERVER_URL` | `http://localhost:8000/mcp` | Public URL MeetStream uses to reach this server. Also settable in Settings → Meetings → Public address; the environment variable wins |
| `APP_HOST` | `127.0.0.1` | Bind address; `0.0.0.0` to accept connections from other machines |
| `CORS_ORIGINS` | `[]` | Extra origins allowed to call the API with a session cookie |
| `SESSION_SECRET` | generated | Cookie signing key; generated into `data/session.key` on first run |
| `TRUST_PROXY` | `false` | Read client address/scheme from `X-Forwarded-*` — only behind your own reverse proxy |
| `API_DOCS` | dev only | Interactive API docs at `/docs` |
| `ALLOW_SELF_SIGNUP` | `true` | Let people create their own account (new workspace, or join with a code). Set `false` on a public server so only owners can add members; the first account is always allowed |

Configuration saved from the UI lives in `data/config.json` next to the SQLite
database, alongside `session.key`. All of `data/` is gitignored — it holds API
keys.

## Docker

```bash
docker compose up -d                       # app + SQLite, http://localhost:8000
docker compose --profile postgres up -d    # app + a pgvector Postgres to pick in Settings
```

One image contains the API and the built UI; everything it writes goes to the
`data` volume. Set `LLM_PROVIDER`/`LLM_API_KEY` (and friends) in
`docker-compose.yml` to skip onboarding entirely.
