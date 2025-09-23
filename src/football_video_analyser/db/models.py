"""SQLAlchemy models for football video analytics."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    func,
)
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .config import get_db_schema

SCHEMA = get_db_schema()
TABLE_KWARGS = {"schema": SCHEMA} if SCHEMA else {}


convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base with future-style SQLAlchemy support."""

    metadata = MetaData(naming_convention=convention)


ID_COLUMN_TYPE = BigInteger().with_variant(Integer, "sqlite")


class MatchVideo(Base):
    """Represents a single match recording available for analysis."""

    __tablename__ = "match_video"
    __table_args__ = TABLE_KWARGS

    id: Mapped[int] = mapped_column(ID_COLUMN_TYPE, primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    fps: Mapped[float] = mapped_column(Float, nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    filesize_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bitrate_kbps: Mapped[float | None] = mapped_column(Float)
    recorded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=datetime.utcnow,
    )

    events: Mapped[list["EventAnnotation"]] = relationship(
        back_populates="match_video",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"MatchVideo(id={self.id!r}, path={self.path!r}, resolution={self.width}x{self.height}, "
            f"fps={self.fps}, duration={self.duration_seconds})"
        )


class EventAnnotation(Base):
    """Individual annotated event within a match video."""

    __tablename__ = "event_annotation"
    __table_args__ = TABLE_KWARGS

    id: Mapped[int] = mapped_column(ID_COLUMN_TYPE, primary_key=True, autoincrement=True)
    match_video_id: Mapped[int] = mapped_column(
        ID_COLUMN_TYPE,
        ForeignKey(f"{SCHEMA + '.' if SCHEMA else ''}match_video.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    team: Mapped[str | None] = mapped_column(String(64))
    player_label: Mapped[str | None] = mapped_column(String(128))
    confidence: Mapped[float | None] = mapped_column(Float)
    extra_data: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=datetime.utcnow,
    )

    match_video: Mapped[MatchVideo] = relationship(back_populates="events")


__all__ = ["Base", "MatchVideo", "EventAnnotation", "SCHEMA"]
