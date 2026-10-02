"""
Alembic wiring: the baseline builds the schema, adoption stamps an existing
pre-Alembic database, and the models have not drifted from the baseline.
"""
import tempfile
import uuid
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.database.bootstrap import ensure_schema
from app.models.database import Base

APP_TABLES = set(Base.metadata.tables)


def _head_revision() -> str:
    """Whatever the newest revision is, so this file needs no edit per migration."""
    from pathlib import Path

    from alembic.config import Config
    from alembic.script import ScriptDirectory

    root = Path(__file__).resolve().parents[1]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "app" / "migrations"))
    return ScriptDirectory.from_config(cfg).get_current_head()


def _fresh_url() -> str:
    path = Path(tempfile.mkdtemp()) / f"mig-{uuid.uuid4().hex}.db"
    return f"sqlite+aiosqlite:///{path.as_posix()}"


async def _tables(engine) -> set:
    async with engine.connect() as conn:
        return set(await conn.run_sync(lambda c: inspect(c).get_table_names()))


async def _version(engine) -> str | None:
    async with engine.connect() as conn:
        if not await conn.run_sync(lambda c: inspect(c).has_table("alembic_version")):
            return None
        return (await conn.execute(text("select version_num from alembic_version"))).scalar()


@pytest.mark.asyncio
async def test_fresh_database_is_migrated_to_head():
    engine = create_async_engine(_fresh_url())
    try:
        await ensure_schema(engine)
        tables = await _tables(engine)
        assert APP_TABLES <= tables, APP_TABLES - tables
        assert await _version(engine) == _head_revision()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_existing_prealembic_database_is_adopted_not_rebuilt():
    engine = create_async_engine(_fresh_url())
    try:
        # Simulate a pre-Alembic install: schema exists, no alembic_version.
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        assert await _version(engine) is None
        await ensure_schema(engine)
        assert await _version(engine) == _head_revision()  # stamped, not re-run
        assert APP_TABLES <= await _tables(engine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_dropped_tables_under_stale_version_are_rebuilt():
    engine = create_async_engine(_fresh_url())
    try:
        await ensure_schema(engine)  # now at head
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)  # leaves alembic_version
        await ensure_schema(engine)  # must notice tables gone and rebuild
        assert APP_TABLES <= await _tables(engine)
        assert await _version(engine) == _head_revision()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_models_have_not_drifted_from_the_baseline():
    """
    Autogenerate against a database at head must find nothing to do; a real
    schema change would show up here and remind the author to add a revision.
    """
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    engine = create_async_engine(_fresh_url())
    try:
        await ensure_schema(engine)
        async with engine.connect() as conn:
            def _compare(sync_conn):
                ctx = MigrationContext.configure(
                    sync_conn,
                    opts={"compare_type": True, "target_metadata": Base.metadata},
                )
                return compare_metadata(ctx, Base.metadata)
            diff = await conn.run_sync(_compare)
    finally:
        await engine.dispose()
    # On SQLite the diff should be empty (pgvector HNSW indexes never appear here).
    assert diff == [], f"models drifted from baseline; add a migration: {diff}"


@pytest.mark.asyncio
async def test_prealembic_accounts_get_a_membership_on_upgrade():
    """
    Adoption is create_all + stamp head, so migration 0002's membership
    backfill never runs for a v0.2.0 database. Its accounts then had no
    workspace at all - an empty list the UI could not render. Bootstrap
    must mirror users.organization_id/role into memberships itself.
    """
    from datetime import datetime, timezone

    from app.database.bootstrap import bootstrap
    from app.models.database import Membership

    engine = create_async_engine(_fresh_url())
    try:
        # A v0.2.0 database: every table except memberships, no alembic_version,
        # one workspace with one owner who predates memberships entirely.
        old_tables = [t for name, t in Base.metadata.tables.items() if name != "memberships"]
        async with engine.begin() as conn:
            await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=old_tables))
            org_id, user_id = uuid.uuid4(), uuid.uuid4()
            now = datetime.now(timezone.utc).isoformat()
            await conn.execute(text(
                "INSERT INTO organizations (id, name, slug, settings, mcp_token, created_at, updated_at) "
                "VALUES (:id, 'Old', 'old', '{}', 'tok', :now, :now)"
            ), {"id": org_id.hex, "now": now})
            await conn.execute(text(
                "INSERT INTO users (id, organization_id, email, name, role, is_active, settings, password_hash, created_at, updated_at) "
                "VALUES (:id, :org, 'old@example.com', 'Old', 'owner', 1, '{}', 'x', :now, :now)"
            ), {"id": user_id.hex, "org": org_id.hex, "now": now})

        await bootstrap(engine)

        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
        from sqlalchemy import select
        async with async_sessionmaker(bind=engine, class_=AsyncSession)() as session:
            rows = (await session.execute(select(Membership))).scalars().all()
        assert [(m.user_id, m.organization_id, m.role) for m in rows] == [(user_id, org_id, "owner")]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_an_old_notes_table_gets_edited_by_user_and_keeps_its_edits():
    """
    Adoption used to stamp head, so a revision that adds a column to an
    existing table never ran on a pre-Alembic database. A meeting's note a
    person had edited (updated well after it was written) stays protected.
    """
    from datetime import datetime, timedelta, timezone

    from app.database.bootstrap import bootstrap

    engine = create_async_engine(_fresh_url())
    try:
        async with engine.begin() as conn:
            await conn.run_sync(lambda c: Base.metadata.create_all(c))
            await conn.execute(text("ALTER TABLE notes DROP COLUMN edited_by_user"))
            org_id, meeting_id = uuid.uuid4(), uuid.uuid4()
            written = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
            await conn.execute(text(
                "INSERT INTO organizations (id, name, slug, settings, mcp_token, created_at, updated_at) "
                "VALUES (:id, 'Old', 'old', '{}', 'tok', :now, :now)"
            ), {"id": org_id.hex, "now": written.isoformat()})
            await conn.execute(text(
                "INSERT INTO meetings (id, organization_id, title, status, processing_status, custom_attributes, created_at, updated_at) "
                "VALUES (:id, :org, 'Sync', 'completed', 'completed', '{}', :now, :now)"
            ), {"id": meeting_id.hex, "org": org_id.hex, "now": written.isoformat()})
            for title, updated in (("edited", written + timedelta(hours=2)), ("untouched", written)):
                await conn.execute(text(
                    "INSERT INTO notes (id, organization_id, meeting_id, title, content, note_type, tags, is_favorite, created_at, updated_at) "
                    "VALUES (:id, :org, :meeting, :title, 'x', 'meeting', '[]', 0, :created, :updated)"
                ), {"id": uuid.uuid4().hex, "org": org_id.hex, "meeting": meeting_id.hex, "title": title,
                    "created": written.isoformat(), "updated": updated.isoformat()})

        await bootstrap(engine)

        assert await _version(engine) == _head_revision()
        async with engine.connect() as conn:
            rows = dict((await conn.execute(text("SELECT title, edited_by_user FROM notes"))).all())
        assert {title: bool(flag) for title, flag in rows.items()} == {"edited": True, "untouched": False}
    finally:
        await engine.dispose()
