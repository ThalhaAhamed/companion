"""notes.edited_by_user: a person's edit is recorded, not inferred from timestamps

Revision ID: 0004_note_edited_by_user
Revises: 0003_membership_status
Create Date: 2026-10-02

A meeting's note counted as edited when updated_at was more than two seconds
after created_at. An edit inside those two seconds was missed, so Reprocess
regenerated the note over it; favouriting or moving a note counted as an
edit, so a fresh summary never reached it. The flag is now set when the
title or text actually changes. Existing notes keep the old reading.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_note_edited_by_user"
down_revision: Union[str, None] = "0003_membership_status"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _as_datetime(value):
    # SQLite hands back text, Postgres a datetime.
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def upgrade() -> None:
    bind = op.get_bind()
    # 0001 builds "notes" from today's model, so a brand-new database already
    # has the column; only an existing one needs it added.
    if "edited_by_user" not in {c["name"] for c in sa.inspect(bind).get_columns("notes")}:
        op.add_column(
            "notes",
            sa.Column("edited_by_user", sa.Boolean(), nullable=False, server_default=sa.false()),
        )
    rows = bind.execute(sa.text("SELECT id, created_at, updated_at FROM notes WHERE meeting_id IS NOT NULL")).all()
    edited = []
    for note_id, created, updated in rows:
        created, updated = _as_datetime(created), _as_datetime(updated)
        if created and updated and updated.replace(tzinfo=None) - created.replace(tzinfo=None) > timedelta(seconds=2):
            edited.append(note_id)
    for note_id in edited:
        bind.execute(sa.text("UPDATE notes SET edited_by_user = :yes WHERE id = :id"), {"yes": True, "id": note_id})


def downgrade() -> None:
    with op.batch_alter_table("notes") as batch:
        batch.drop_column("edited_by_user")
