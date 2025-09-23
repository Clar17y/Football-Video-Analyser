"""Add event annotation table"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from football_video_analyser.db.config import get_db_schema

# revision identifiers, used by Alembic.
revision = "0002_add_event_annotation"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None

SCHEMA = get_db_schema()
SCHEMA_KWARGS = {"schema": SCHEMA} if SCHEMA else {}
ID_TYPE = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
MATCH_VIDEO_TABLE = f"{SCHEMA + '.' if SCHEMA else ''}match_video"


def upgrade() -> None:
    op.create_table(
        "event_annotation",
        sa.Column("id", ID_TYPE, primary_key=True, autoincrement=True),
        sa.Column("match_video_id", ID_TYPE, nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("timestamp_seconds", sa.Float(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("team", sa.String(length=64), nullable=True),
        sa.Column("player_label", sa.String(length=128), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("extra_data", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["match_video_id"],
            [f"{MATCH_VIDEO_TABLE}.id"],
            ondelete="CASCADE",
        ),
        **SCHEMA_KWARGS,
    )

    op.create_index(
        "ix_event_annotation_match_video_id_ts",
        "event_annotation",
        ["match_video_id", "timestamp_seconds"],
        unique=False,
        **SCHEMA_KWARGS,
    )


def downgrade() -> None:
    op.drop_index("ix_event_annotation_match_video_id_ts", "event_annotation", **SCHEMA_KWARGS)
    op.drop_table("event_annotation", **SCHEMA_KWARGS)
