# Troubleshooting

**A meeting sits at "Joining" or "Left the call" for a long time.** The
server asks MeetStream about the bot every 20 seconds, so the status should
follow the call within half a minute and extraction should start on its own
once MeetStream has the transcript (a few minutes after the call). If it
does not move at all, the server cannot reach MeetStream (network, or the
API key was changed after launch); the server log says why. *Reprocess*
fetches the transcript by id at any time. The agent answering *in* the call
is the part that needs a public address — **Settings → Meetings → Public
address** (see [Live meetings](live-meetings.md)).

**"Processed without AI (…)"** on a meeting. The AI provider failed and the
rule-based parser ran instead; the message in brackets is the provider's own
reason. For Ollama, `Model 'x' is not pulled` means `ollama pull x`; a CUDA
*out of memory* means the GPU is full - close other GPU work or pick a
smaller model. Fix the provider in Settings, then *Reprocess*.

**Ask AI answers "Could not answer that".** Same cause as above; the error
text is the provider's. *Test connection* in Settings → AI provider reproduces
it without a meeting.

**Blank page after upgrading from v0.2.0.** Fixed in v0.3.1 - accounts from
before workspaces had no membership row. Upgrade; the server repairs it on
start.

**Port 8000 already in use.** Find what holds it (`netstat -ano | findstr :8000`
on Windows, `lsof -i :8000` elsewhere) and stop it - the Vite dev proxy in
`frontend/vite.config.js` expects the API on 8000. Running the API alone on
another port is `scripts/dev.py --port 8010 --no-ui`.

**Database "Not reachable" in Settings.** The banner shows the driver's
reason. *Connection refused* → nothing listening on that host/port;
*could not resolve host* → check the hostname; a Supabase direct URL on a
network without IPv6 → use the pooler connection string instead.

**The desktop app shows the sign-in screen although it used to sign itself in.**
That happens only when the workspace has more than one owner - the device key
signs in *the* owner and refuses to guess between several. Sign in normally.

**Signed out, and your account is on a different database.** On the desktop
app the sign-in page names the database it is about to sign you in to, with
the other saved ones underneath it — pick one and it switches there. If that
database already knows this machine, you land straight in the app; otherwise
you get the sign-in form again, now on the right database. The same menu's
**Connect another database** takes a connection string for one this computer
has never seen, saves it and switches to it - the onboarding wizard's
"connect to your team's database" step, reachable after onboarding is done.

**"Incorrect email or password" right after switching databases.** Accounts
live in the database, so on a database that already has people in it your
old account does not exist — sign in with an account from *that* database,
or ask its owner to add you. On an *empty* database (a fresh Postgres) your
account is carried over automatically as of v0.4.3 and you stay signed in;
your meetings and notes are not copied — they remain in the previous
database. To bring them across, see `scripts/import_from_postgres.py`.
