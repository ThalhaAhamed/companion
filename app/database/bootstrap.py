"""
Bring a database up to date and ready for first use.

Runs at startup and again whenever the database is switched from Settings,
so the same code path serves a brand new SQLite file, an existing Postgres
deployment and a freshly pasted Supabase URL.
"""
from __future__ import annotations

import logging
import secrets
import uuid
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.config import settings
from app.models.database import Base
from app.secrets import configured_mcp_token

POSTGRESQL = "postgresql"

#: MCP tokens that were ever shipped as defaults. See app/secrets.py.
PLACEHOLDER_TOKENS = frozenset({"dev-mcp-token-meetstream-2026", "local-dev-token", "change-me-to-a-random-string"})


#: Advisory-lock key for start-up schema/seed work. Arbitrary, but must be the
#: same in every process sharing a database.
BOOTSTRAP_LOCK_KEY = 8_274_301_556_120_733


async def bootstrap(engine: AsyncEngine) -> None:
    """
    Schema first, then the rows every request depends on.

    Serialized with a Postgres advisory lock, because a shared database is a
    supported deployment: several people each run the app on their own machine
    pointed at one Postgres. Booting two of them at the same moment used to
    race - both created alembic_version, both inserted the default workspace -
    and the loser died with "duplicate key ... already exists" before it ever
    served a request. Whoever gets the lock does the work; the others wait a
    moment and then find nothing left to do.

    SQLite needs none of this: it is a single local file, not a shared server.
    """
    async with _bootstrap_lock(engine):
        await ensure_schema(engine)
        await ensure_default_workspace(engine)
        await ensure_workspace_owners(engine)
        await ensure_memberships(engine)
        await prune_operational_tables(engine)
        await fail_interrupted_processing(engine)


@asynccontextmanager
async def _bootstrap_lock(engine: AsyncEngine):
    if engine.dialect.name != POSTGRESQL:
        yield
        return

    # AUTOCOMMIT: a session-level advisory lock must not be tied to a
    # transaction that the migration work below might roll back.
    conn = await engine.connect()
    try:
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        await conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": BOOTSTRAP_LOCK_KEY})
        try:
            yield
        finally:
            await conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": BOOTSTRAP_LOCK_KEY})
    finally:
        await conn.close()


async def ensure_schema(engine: AsyncEngine) -> None:
    """
    Bring the database up to date at startup, through Alembic.

    Three cases, so the zero-setup promise holds and existing installs are not
    disturbed:

    * Brand-new database (no application tables): run every migration. A local
      SQLite file still needs no psql session and no manual step - the app
      creates and migrates it on first run.
    * Existing database created before Alembic (tables but no alembic_version):
      apply the historical idempotent patches once to close any old-schema
      gaps, then stamp it at the baseline so it becomes migration-managed.
    * Already migration-managed: upgrade to head, applying any new revisions.

    pgvector's extension and its HNSW indexes are not expressible in portable
    ORM metadata, so they are (re)created here after the schema is in place.
    """
    dialect = engine.dialect.name

    async with engine.connect() as conn:
        has_app_tables = await conn.run_sync(_has_table, "organizations")
        has_alembic = await conn.run_sync(_has_table, "alembic_version")

    if not has_app_tables:
        # Fresh database - or one whose app tables were dropped out from under
        # a stale alembic_version (e.g. a per-test reset). A version row that
        # claims "head" would make `upgrade` a no-op and leave no schema, so
        # clear it first and build from the baseline.
        if has_alembic:
            async with engine.begin() as conn:
                await conn.execute(text("DROP TABLE alembic_version"))
        await _alembic(engine, "upgrade", "head")
    elif not has_alembic:
        # Adopt an existing pre-Alembic database: make sure its schema is fully
        # current, then hand ownership to Alembic without re-running the baseline.
        async with engine.begin() as conn:
            if dialect == POSTGRESQL:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.run_sync(Base.metadata.create_all)
                await _patch_legacy_postgres_schema(conn)
            else:
                await conn.run_sync(Base.metadata.create_all)
            await _patch_action_items(conn, dialect)
            await conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_users_email ON users (email)"))
        # create_all above built the missing tables as the models describe
        # them *today*, but left the existing ones as they were. Stamping
        # baseline would queue revisions that create tables which now exist;
        # stamping head skipped revisions that add columns to old tables (a
        # pre-Alembic notes table never got edited_by_user). So: stamp the
        # last revision create_all fully covers, then upgrade - revisions
        # after it check for what they add before adding it.
        await _alembic(engine, "stamp", ADOPTED_AT_REVISION)
        await _alembic(engine, "upgrade", "head")
    else:
        # Already migration-managed: apply any new revisions.
        await _alembic(engine, "upgrade", "head")

    # Extension + vector indexes live outside the ORM metadata (and outside the
    # migrations, since the HNSW/ivfflat choice has runtime logic); ensure them
    # every start. Cheap and idempotent.
    if dialect == POSTGRESQL:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await _create_postgres_vector_indexes(conn)


#: The first migration.
BASELINE_REVISION = "0001_baseline"
#: Where an adopted pre-Alembic database is stamped before upgrading: every
#: revision up to here only touched tables such a database did not have,
#: which create_all has just built whole.
ADOPTED_AT_REVISION = "0003_membership_status"


def _has_table(sync_conn, name: str) -> bool:
    from sqlalchemy import inspect

    return inspect(sync_conn).has_table(name)


async def _alembic(engine: AsyncEngine, action: str, target: str) -> None:
    """
    Run an Alembic command against this engine's URL.

    Alembic's env.py runs its own asyncio loop, so it is executed in a worker
    thread to avoid nesting event loops inside the startup lifespan. The URL is
    passed through config attributes so env.py targets exactly this database.
    """
    import asyncio
    import sys
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    # In the PyInstaller bundle, alembic.ini and app/migrations are unpacked
    # under _MEIPASS; from source they sit at the repository root.
    if getattr(sys, "frozen", False):
        root = Path(sys._MEIPASS)
    else:
        root = Path(__file__).resolve().parents[2]

    def _run() -> None:
        cfg = Config(str(root / "alembic.ini"))
        cfg.set_main_option("script_location", str(root / "app" / "migrations"))
        # str(engine.url) masks the password as "***"; render it in full so
        # Alembic can actually authenticate (e.g. Postgres).
        cfg.attributes["url"] = engine.url.render_as_string(hide_password=False)
        if action == "upgrade":
            command.upgrade(cfg, target)
        elif action == "stamp":
            command.stamp(cfg, target)
        else:  # pragma: no cover - guard
            raise ValueError(action)

    await asyncio.to_thread(_run)


async def _patch_action_items(conn, dialect: str) -> None:
    """
    Action items used to require a meeting; hand-written tasks in notes
    do not have one. Adds note_id and relaxes meeting_id on databases created
    before that change. create_all above already did this for new ones.
    """
    if dialect == POSTGRESQL:
        await conn.execute(text(
            "ALTER TABLE action_items ADD COLUMN IF NOT EXISTS note_id UUID REFERENCES notes(id) ON DELETE SET NULL"
        ))
        await conn.execute(text("ALTER TABLE action_items ALTER COLUMN meeting_id DROP NOT NULL"))
        return

    if dialect != "sqlite":
        return

    columns = {row[1]: row for row in (await conn.execute(text("PRAGMA table_info(action_items)"))).all()}
    if "note_id" not in columns:
        await conn.execute(text("ALTER TABLE action_items ADD COLUMN note_id CHAR(32) REFERENCES notes(id) ON DELETE SET NULL"))
        columns = {row[1]: row for row in (await conn.execute(text("PRAGMA table_info(action_items)"))).all()}
    # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
    if columns["meeting_id"][3]:
        # SQLite cannot drop NOT NULL in place: rebuild the table from the
        # current model and copy the rows across.
        table = Base.metadata.tables["action_items"]
        names = ", ".join(column.name for column in table.columns)
        await conn.execute(text("PRAGMA foreign_keys=OFF"))
        await conn.execute(text("ALTER TABLE action_items RENAME TO action_items__old"))
        # Indexes follow the renamed table and would collide with the ones
        # table.create() declares.
        for index in table.indexes:
            await conn.execute(text(f"DROP INDEX IF EXISTS {index.name}"))
        await conn.run_sync(lambda sync_conn: table.create(sync_conn))
        await conn.execute(text(f"INSERT INTO action_items ({names}) SELECT {names} FROM action_items__old"))
        await conn.execute(text("DROP TABLE action_items__old"))
        await conn.execute(text("PRAGMA foreign_keys=ON"))


async def _create_postgres_vector_indexes(conn):
    """
    Create the HNSW indexes that back similarity search on Postgres.

    These cannot be declared on the ORM models because the index type only
    exists in pgvector; the portable backend needs no index at all.

    HNSW rather than ivfflat: ivfflat builds its centroids from whatever
    rows exist at CREATE INDEX time, so an index created on an empty table
    (every fresh install) has meaningless lists and poor recall until
    someone reindexes by hand. HNSW builds incrementally and needs no
    training data. Databases that already have the old ivfflat index keep
    working; the ivfflat one is dropped and replaced the first time a new
    version starts.
    """
    for table in ("meeting_memory_embeddings", "company_knowledge_embeddings", "notes"):
        await conn.execute(text(f"DROP INDEX IF EXISTS idx_{table}_embedding_ivfflat"))
        legacy = await conn.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :name"), {"name": f"idx_{table}_embedding"}
        )
        definition = legacy.scalar_one_or_none() or ""
        if "ivfflat" in definition:
            await conn.execute(text(f"DROP INDEX IF EXISTS idx_{table}_embedding"))
        await conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_embedding "
                f"ON {table} USING hnsw (embedding vector_cosine_ops)"
            )
        )


async def _patch_legacy_postgres_schema(conn):
    """
    Additive patches for Postgres databases created before these columns
    existed. Guarded with IF NOT EXISTS so they are safe to re-run, and skipped
    entirely on databases created from current metadata.
    """
    await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255)"))
    await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE"))
    await conn.execute(text("ALTER TABLE organizations ADD COLUMN IF NOT EXISTS mcp_token VARCHAR(255) UNIQUE"))
    await conn.execute(text("ALTER TABLE organizations ADD COLUMN IF NOT EXISTS join_code VARCHAR(50) UNIQUE"))
    # The existing default-org workspace's live MeetStream agent is already
    # wired with the single global MCP_AUTH_TOKEN from before per-workspace
    # tokens existed - backfill it as that org's own mcp_token so its agent
    # keeps working without needing to be re-wired.
    explicit_token = configured_mcp_token()
    if explicit_token:
        await conn.execute(
            text(
                "UPDATE organizations SET mcp_token = :token "
                "WHERE id = :org_id AND mcp_token IS NULL"
            ),
            {"token": explicit_token, "org_id": settings.DEFAULT_ORG_ID},
        )
    await conn.execute(
        text(
            "UPDATE organizations SET join_code = substr(md5(random()::text), 1, 8) "
            "WHERE join_code IS NULL"
        )
    )
    # One-time cleanup: member removal used to set is_active=FALSE instead
    # of deleting the row (fixed in app/api/members.py), so a removed
    # member's email stayed permanently reserved by the (organization_id,
    # email) constraint - "that email is already a member" for an email
    # nobody could actually sign back in with. Removal deletes the row now,
    # so this only ever needs to run once for rows soft-deleted before
    # that fix.
    await conn.execute(text("DELETE FROM users WHERE is_active = FALSE"))

    await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS settings JSONB DEFAULT '{}'::jsonb"))
    # Tracks which member's own MeetStream API key (and account) a bot was
    # actually deployed under, so status/stop calls on that bot later use
    # the same key it was created with - necessary now that each member
    # can configure their own key (see app/api/agent.py get_meetstream_api_key),
    # since a bot created under one member's MeetStream account can only
    # be queried/stopped with that same account's key, not another
    # member's. Existing rows stay NULL (falls back to the deployment's
    # shared default key, same as before this column existed).
    await conn.execute(text("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS created_by_user_id UUID REFERENCES users(id) ON DELETE SET NULL"))
    # Agent ownership/activation moved from the shared workspace to each
    # individual member - meeting memory stays workspace-shared, but who
    # you're talking to as "your agent" doesn't. Backfill: members of the
    # original default workspace inherit whatever agent the workspace had
    # active before this change, as their own personal starting point,
    # rather than suddenly having no agent configured at all.
    await conn.execute(
        text(
            """
            UPDATE users u
            SET settings = jsonb_build_object(
                'active_agent_config_id', o.settings->'active_agent_config_id',
                'agent_config_ids', COALESCE(o.settings->'agent_config_ids', '[]'::jsonb)
            )
            FROM organizations o
            WHERE u.organization_id = o.id
              AND u.organization_id = :org_id
              AND NOT (u.settings ? 'active_agent_config_id')
            """
        ),
        {"org_id": settings.DEFAULT_ORG_ID},
    )


async def ensure_default_workspace(engine: AsyncEngine) -> None:
    """
    Guarantee the default workspace exists.

    A fresh install - a new SQLite file in particular - has no rows at all, and
    every request resolves through an organization. Creating it here is what
    lets the application come up usable on first run without a seeding script.
    """
    from app.models.database import Organization

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        org_id = uuid.UUID(settings.DEFAULT_ORG_ID)
        existing = (
            await session.execute(select(Organization).where(Organization.id == org_id))
        ).scalar_one_or_none()

        if existing is None:
            session.add(
                Organization(
                    id=org_id,
                    name="My Workspace",
                    slug="default",
                    settings={},
                    mcp_token=configured_mcp_token() or secrets.token_urlsafe(32),
                    join_code=secrets.token_hex(4),
                )
            )
        else:
            # Older databases may predate these columns having values.
            if not existing.mcp_token:
                existing.mcp_token = configured_mcp_token() or secrets.token_urlsafe(32)
            elif existing.mcp_token in PLACEHOLDER_TOKENS:
                # Installs created before tokens were generated shipped with a
                # token that is public in this repository. Replace it; the
                # agent, if any, is re-wired on next activation.
                existing.mcp_token = secrets.token_urlsafe(32)
            if not existing.join_code:
                existing.join_code = secrets.token_hex(4)
        await session.commit()




async def ensure_workspace_owners(engine: AsyncEngine) -> None:
    """
    Every workspace has at least one owner.

    Roles predate this code: existing members are all "member". The earliest
    active member of each workspace without an owner becomes its owner, which
    is who created it in every realistic history.
    """
    from app.models.database import User

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        owned = {
            row[0]
            for row in (
                await session.execute(
                    select(User.organization_id).where(User.role == "owner", User.is_active.is_(True)).distinct()
                )
            ).all()
        }
        candidates = (
            await session.execute(
                select(User).where(User.is_active.is_(True)).order_by(User.organization_id, User.created_at)
            )
        ).scalars().all()
        changed = False
        for user in candidates:
            if user.organization_id in owned:
                continue
            user.role = "owner"
            owned.add(user.organization_id)
            changed = True
        if changed:
            await session.commit()


async def ensure_memberships(engine: AsyncEngine) -> None:
    """
    Every account belongs to at least the workspace on its user row.

    Migration 0002 backfills this, but a database whose memberships table
    was created by create_all and then stamped never ran that backfill -
    and an account with no membership at all gets an empty workspace list,
    which the UI could not render. Mirror users.organization_id/role for
    anyone still missing a row.
    """
    from app.models.database import Membership, User

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        with_rows = {row[0] for row in (await session.execute(select(Membership.user_id).distinct())).all()}
        orphans = (
            await session.execute(
                select(User).where(User.is_active.is_(True), User.organization_id.is_not(None))
            )
        ).scalars().all()
        added = 0
        for user in orphans:
            if user.id in with_rows:
                continue
            session.add(Membership(user_id=user.id, organization_id=user.organization_id, role=user.role or "member"))
            added += 1
        if added:
            await session.commit()
            logger.info("Backfilled %d missing workspace membership(s)", added)


#: Webhook deliveries and processing-job records are diagnostics, not data.
#: Older rows than this are deleted at startup so the tables cannot grow
#: without bound on a long-running install.
OPERATIONAL_RETENTION_DAYS = 30


async def prune_operational_tables(engine: AsyncEngine) -> None:
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import delete

    from app.models.database import ProcessingJob, WebhookEvent

    cutoff = datetime.now(timezone.utc) - timedelta(days=OPERATIONAL_RETENTION_DAYS)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        await session.execute(delete(WebhookEvent).where(WebhookEvent.received_at < cutoff, WebhookEvent.processed.is_(True)))
        await session.execute(delete(ProcessingJob).where(ProcessingJob.created_at < cutoff, ProcessingJob.status != "running"))
        await session.commit()


async def fail_interrupted_processing(engine: AsyncEngine) -> None:
    """
    Extraction runs as an in-process task, so a restart mid-way leaves the
    meeting saying "processing" with nobody working on it - and the UI
    spinning forever. At boot nothing can be in flight, so anything still
    marked queued/processing was interrupted: say so, and let the user
    reprocess.
    """
    from sqlalchemy import update

    from app.models.database import Meeting

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        result = await session.execute(
            update(Meeting)
            .where(Meeting.processing_status.in_(("queued_for_processing", "processing")))
            .values(
                processing_status="failed",
                processing_error="Processing was interrupted by a server restart. Use Reprocess to run it again.",
            )
        )
        await session.commit()
        if result.rowcount:
            logger.warning("Marked %d interrupted meeting(s) as failed after restart", result.rowcount)
