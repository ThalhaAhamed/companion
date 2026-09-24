# Live meetings: reaching your server

Notes, search, Ask AI and transcript upload work entirely on your machine.
**Sending a bot into a live call needs MeetStream to reach your server** for two
things: webhook deliveries (`/api/webhooks/meetstream`) and the voice agent's
tool calls (`/mcp`).

This has nothing to do with which database you use — SQLite, Supabase or
any other Postgres only changes where data is stored; MeetStream never
talks to the database, it talks to the Meet Companion server. It is about
where the *server* runs: on a host with a public address (a VPS, Railway,
Fly, …) no tunnel is needed at all; on a laptop, some tunnel is unavoidable
while a bot is in a call.

**On a laptop, the easy way:** Settings → Meetings → Public address → *Start
a tunnel automatically*. It is switched on for you when an owner saves a
MeetStream API key (onboarding or Settings), unless an address is already
set or you have switched it on or off yourself. Meet Companion runs a Cloudflare quick tunnel
(`cloudflared` ships in the desktop installer, checksum-pinned) for as long
as it is open, waits until the address answers, and uses it — nothing to
install, run or paste. Only MeetStream's paths answer through it (`/mcp`,
`/api/webhooks/…`, the chat relay and `/health`); sign-in, the UI and the
API return 404 there, so your install is not put on the internet. The
address changes whenever the tunnel restarts; bots always get the current
one. Quick tunnels come with no uptime guarantee, so for a team server
prefer a fixed address (below).

**Or set an address yourself:**

1. Expose the server on a public URL — a real domain behind HTTPS, or during
   development a tunnel such as `cloudflared tunnel --url http://localhost:8000`
   or `ngrok http 8000`. Cloudflare's free *quick* tunnels
   (`*.trycloudflare.com`) work for webhooks, `/mcp` and updating an existing
   agent, but MeetStream's agent-*creation* endpoint returns a 500 when a
   `custom_functions` URL is on that domain — so create the agent while on
   another host, or use a named Cloudflare tunnel / ngrok. Quick-tunnel URLs
   also change on every launch; for anything beyond a one-off test use a
   fixed domain.
2. Enter that address in **Settings → Meetings → Public address**
   (`https://<that-host>` is enough) and press **Save and check**: it is
   probed through the address itself and says whether MeetStream can reach
   this server. No restart; the next bot you launch uses it. The webhook
   callback URL is derived from it. On a server you can set
   `MCP_SERVER_URL=https://<that-host>/mcp` in the environment instead,
   which takes precedence and locks the field.
3. Add your MeetStream API key in **Settings → Meetings** and a webhook
   signing secret (the same value on both sides).
4. Create or activate an agent in **Agent**; Meet Companion wires the MCP
   server URL, the `share_in_chat` chat relay and your workspace's token
   into it, and re-checks that wiring every time you launch a bot, so a
   tunnel that came back on a new address is picked up without doing
   anything. If the agent *cannot* reach this server — no public https
   address, or a tunnel that has closed — the Agent page and the launch
   dialog say so before the call, instead of the agent quietly answering
   every memory question with a guess.

Without a reachable URL a bot still joins and records, and the meeting still
follows it: the server asks MeetStream for the bot's state every 20 seconds
while a call is live (Joining → In the call → Recording → Left the call), and
once MeetStream has finished the transcript it fetches it and runs the
summary, memory and action-item extraction by itself — the webhooks are the
faster path, not the only one. What a tunnel-less install does lose is the
agent: it cannot reach memory during the call. A call in which nobody speaks
produces no transcript on MeetStream's side; the meeting is marked failed
with that reason.

## Getting quick, accurate answers

Activating an agent (**Use this agent**) also tunes it for a live call.
For Gemini's native-audio model it turns off the "thinking" pass that ran
before every spoken reply, and turns Gemini's own end-of-speech detection
back on when it had been switched off without the external transcriber
MeetStream requires for that — without either, the agent was slow to
notice you had finished talking. Memory lookups time out after 10 seconds,
MeetStream's default. Other model choices (voice, temperature, provider)
stay as you set them.

The introduction message is posted into the meeting chat by Meet Companion
**once the bot has been admitted**, not when it first joins: on Google Meet
a bot joins into the waiting room, where it cannot reach the chat, and a
message sent then was lost.

## Answering by voice or in the chat

On the **Agent** page each agent has a *How it answers in a call* setting:

| Mode | In the call |
| --- | --- |
| **Voice** (default) | The agent answers out loud when someone says its name. |
| **Chat** | The agent stays silent; when someone says its name, the answer is posted in the meeting chat. |

In both modes the agent hears the room and is addressed by name — "Ada,
what did we decide last time?" — and answers from the workspace through the
MCP server as before. The setting is MeetStream's `response_modality` on the
agent (`audio` or `chat`), so it applies to every call that agent joins, from
any install.

**Questions typed into the meeting chat cannot be answered.** MeetStream's
agents do not read the chat panel, no webhook carries chat messages, and the
`get_chats` endpoint only returns the chat after the call has ended (verified
against a live meeting). If MeetStream adds a live chat feed, a typed mode
becomes possible; until then, ask by voice.
