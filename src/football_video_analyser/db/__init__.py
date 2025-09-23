"""Database utilities for football video analysis."""

from .config import DEFAULT_DB_URL, DEFAULT_SCHEMA, get_db_schema, get_db_url
from .ingest import upsert_match_videos
from .models import Base, EventAnnotation, MatchVideo, SCHEMA
from .session import get_engine, get_session_factory

__all__ = [
    "Base",
    "MatchVideo",
    "EventAnnotation",
    "SCHEMA",
    "DEFAULT_DB_URL",
    "DEFAULT_SCHEMA",
    "get_db_url",
    "get_db_schema",
    "get_engine",
    "get_session_factory",
    "upsert_match_videos",
]
