# Security policy

## Reporting a vulnerability

Please do **not** open a public issue for security problems.

Report it privately through
[GitHub's private vulnerability reporting](https://github.com/meetstream-ai/companion/security/advisories/new),
or email **support@meetstream.ai** with "Security" in the subject. Include a
description, steps to reproduce, and the version or commit you tested. You will get an acknowledgement within a few
days. Fixes are released as soon as they are ready and credited in the release
notes unless you prefer otherwise.

## What is in scope

- The API server (`app/`), the web UI (`frontend/`), the desktop shell
  (`desktop/`), the MCP tool server (`/mcp`) and the webhook receiver.
- The default configuration produced by first-run onboarding.

## Deployment model and trust boundaries

Meet Companion is designed to be self-hosted by one team. Understand these
boundaries before exposing an instance beyond localhost:

- **Server-wide settings belong to the instance admin.** The instance admin
  is an owner of the first workspace on the server. The AI provider, the
  database, the MeetStream credentials, the tunnel and the agent template are
  server-wide settings. Only the instance admin can change them. The server
  records the first workspace once. The role never moves to another
  workspace. If that workspace has no active owner left, or is deleted,
  nobody is instance admin. To recover, re-activate or promote an owner of
  that workspace. Or remove the `instance_admin_org_id` key by hand from the
  `settings` of the default workspace row (id
  `00000000-0000-0000-0000-000000000001`) in the database. The next check
  then records the oldest workspace that has an active owner.
- **Workspace owners manage only their own workspace.** The person who creates
  a workspace is its owner and can promote others. An owner of a later
  workspace cannot change server-wide settings.
- **Members of a workspace see all of that workspace's meetings, notes and
  action items.** There is no per-note sharing.
- **Anyone who can reach the server can create a new workspace** (sign-up is
  open) unless you put it behind your own authentication proxy. A new workspace
  never sees another workspace's data. With `ALLOW_SELF_SIGNUP=false`, only the
  instance admin can create more workspaces.
- **The in-call voice agent has write tools** (add notes, create and update
  action items) scoped to its workspace, driven by what is said in the meeting.
  Treat what participants say as untrusted input to those tools - or switch
  them off (Settings → Meetings → *In-call agent permissions*, owners only)
  to keep the agent read-only.
- **Transcripts are sent to the LLM provider you choose.** With a hosted
  provider that means the vendor receives the full text of your meetings.
- **The server fetches the AI base URL that a workspace owner types.** It
  never connects to link-local or cloud metadata addresses (169.254.0.0/16,
  `fd00:ec2::254`, `100.100.100.200`, `metadata.google.internal`). The check
  uses the address that the connection actually goes to, so a DNS change
  after the check has no effect. Private and loopback addresses are allowed
  by default, because teams use internal LLM gateways. Errors from a
  workspace's provider do not include the response body; the install's own
  provider shows only the error message from its JSON reply. A saved key is only sent to the base URL it
  was saved with: a changed base URL needs the key again.

## Hardening checklist

- Run behind HTTPS; the session cookie is marked `Secure`.
- Set `CORS_ORIGINS` to the exact origin of your UI (or leave it empty when the
  server serves the UI itself).
- Set `MEETSTREAM_WEBHOOK_SECRET` (Settings → Meetings, or the environment) so
  webhook deliveries are signature-checked.
- Keep the generated `data/session.key` private; rotating it signs everyone out.
- Bind to `127.0.0.1` (the default) unless the server is meant to be reachable.
- Behind a reverse proxy, set `TRUST_PROXY=true` so rate limiting sees real
  client addresses; never set it when clients connect directly.
- Leave `API_DOCS` off in production (the default outside development).
- If people you do not trust can sign up, set `ALLOW_PRIVATE_LLM_BASE_URL=false`
  so that a workspace AI base URL cannot reach your internal network.

## Known limitations and hardening

Do these things when you expose an instance beyond localhost:

- Set a webhook secret when you use the tunnel or a public URL. Without a
  secret, the server accepts unsigned webhook deliveries for bots that it
  launched.
- Put the server behind a reverse proxy that allows only your own host names.
  The server does not check the `Host` header itself.
- Set `APP_ENV=production` to hide `/docs` and `/openapi.json`. The default
  outside Docker is `development`, which shows them.
