# Configuration

## Settings and environment variables

Everything is configurable from **Settings** in the app. Precedence is:

1. process environment variables (containers, CI) - shown read-only in Settings
2. what you saved in the UI
3. `.env` file (self-hosted convenience)
4. built-in defaults

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite+aiosqlite:///data/meet-companion.db` | Storage; `postgresql+asyncpg://…` for Postgres |
| `LLM_PROVIDER` | `ollama` | `openai` · `anthropic` · `gemini` · `ollama` · `groq` · `openai_compatible` |
| `LLM_MODEL` | per provider | Model name |
| `LLM_API_KEY` | none | Hosted providers only |
| `LLM_BASE_URL` | per provider | Proxies, gateways, self-hosted endpoints |
| `OLLAMA_NUM_CTX` | `16384` (`32768` for larger prompts) | Context window requested from Ollama. Its own default (2048 on Ollama 0.35) silently cuts long prompts from the start; lower this only if the model runs out of memory |
| `MEETSTREAM_API_KEY` | none | Deployment-wide key; each member can also add their own in Settings |
| `MEETSTREAM_WEBHOOK_SECRET` | none | Signature check for webhook deliveries (also settable in Settings) |
| `MCP_SERVER_URL` | `http://localhost:8000/mcp` | Public URL MeetStream uses to reach this server. Also settable in Settings → Meetings → Public address; the environment variable wins |
| `APP_HOST` | `127.0.0.1` | Bind address; `0.0.0.0` to accept connections from other machines |
| `CORS_ORIGINS` | `[]` | Extra origins allowed to call the API with a session cookie |
| `SESSION_SECRET` | generated | Cookie signing key; generated into `data/session.key` on first run |
| `TRUST_PROXY` | `false` | Read client address/scheme from `X-Forwarded-*` - only behind your own reverse proxy |
| `API_DOCS` | dev only | Interactive API docs at `/docs` |
| `ALLOW_SELF_SIGNUP` | `true` | Let people create their own account (new workspace, or join with a code). Set `false` on a public server so only owners can add members; the first account is always allowed |
| `SETUP_TOKEN` | unset | Lock first-run setup and the first account (the server administrator) behind a token. `auto` makes a random token and writes a setup link to the server log; any other value is the token. Open `/?setup_token=<token>` once in the browser you use for setup. Unset: setup is open until the first account exists |
| `APP_BIND` | `127.0.0.1` | Docker Compose only: the host address the port is published on. `0.0.0.0` makes the app reachable from other machines |

Configuration saved from the UI lives in `data/config.json` next to the SQLite
database, alongside `session.key`. All of `data/` is gitignored - it holds API
keys.

## Docker

```bash
docker compose up -d                       # app + SQLite, http://localhost:8000
docker compose --profile postgres up -d    # app + a pgvector Postgres to pick in Settings
```

The port is published on `127.0.0.1` only. Until the first account exists,
setup has no password, and the first account is the server administrator. So
do not publish a new install on a network before setup is done, or lock setup
with a token first:

```bash
SETUP_TOKEN=auto APP_BIND=0.0.0.0 docker compose up -d
docker compose logs app | grep setup_token     # open that link once, in your browser
```

**Upgrading an existing install.** Before this change, Compose published the
port on every interface. If you open the app as `http://<server-ip>:8000`, or
MeetStream reaches it on that port, set `APP_BIND=0.0.0.0` in a `.env` file
next to `docker-compose.yml` before you update, so it stays reachable. A
reverse proxy on the same host (it connects to `127.0.0.1:8000`) and a proxy
container on the compose network need no change.

Another way: keep the default, open an SSH tunnel
(`ssh -L 8000:localhost:8000 you@server`), finish setup at
<http://localhost:8000>, then restart with `APP_BIND=0.0.0.0`. Docker's port
rules bypass host firewalls such as ufw. For a public server, put a reverse
proxy with HTTPS in front and keep `APP_BIND` at `127.0.0.1`.

One image contains the API and the built UI; everything it writes goes to the
`data` volume. Set `LLM_PROVIDER`/`LLM_API_KEY` (and friends) in
`docker-compose.yml` to skip onboarding entirely.
