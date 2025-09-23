"""Configuration helpers for database connections."""

from __future__ import annotations

import os
from functools import lru_cache

DEFAULT_DB_URL = "postgresql+psycopg://user:password@localhost:5432/postgres"
DEFAULT_SCHEMA = "football_video"


@lru_cache(maxsize=1)
def get_db_url() -> str:
    """Return the database URL, defaulting to a placeholder."""

    return os.getenv("FOOTBALL_VIDEO_DB_URL", DEFAULT_DB_URL)


@lru_cache(maxsize=1)
def get_db_schema() -> str:
    """Return the database schema to target."""

    return os.getenv("FOOTBALL_VIDEO_DB_SCHEMA", DEFAULT_SCHEMA)


__all__ = ["get_db_url", "get_db_schema", "DEFAULT_DB_URL", "DEFAULT_SCHEMA"]
