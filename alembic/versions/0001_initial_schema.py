"""Initial football video schema"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from football_video_analyser.db.config import get_db_schema

# revision identifiers, used by Alembic.
revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None

SCHEMA = get_db_schema()
SCHEMA_KWARGS = {"schema": SCHEMA} if SCHEMA else {}


def upgrade() -> None:
    if SCHEMA:
        op.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))

    id_type = sa.BigInteger().with_variant(sa.Integer(), "sqlite")

    op.create_table(
        "match_video",
        sa.Column("id", id_type, primary_key=True, autoincrement=True),
        sa.Column("path", sa.String(length=512), nullable=False, unique=True),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("fps", sa.Float(), nullable=False),
        sa.Column("frame_count", sa.Integer(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("filesize_bytes", sa.BigInteger(), nullable=False),
        sa.Column("bitrate_kbps", sa.Float(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        **SCHEMA_KWARGS,
    )

    op.create_index(
        "ix_match_video_recorded_at",
        "match_video",
        ["recorded_at"],
        unique=False,
        **SCHEMA_KWARGS,
    )


def downgrade() -> None:
    op.drop_index("ix_match_video_recorded_at", "match_video", **SCHEMA_KWARGS)
    op.drop_table("match_video", **SCHEMA_KWARGS)

    if SCHEMA:
        op.execute(sa.text(f'DROP SCHEMA IF EXISTS "{SCHEMA}" CASCADE'))
