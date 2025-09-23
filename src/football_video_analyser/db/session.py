"""Session and engine helpers for the football video database."""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import get_db_url


def get_engine(*, echo: bool = False, **kwargs: Any) -> Engine:
    """Create an SQLAlchemy engine using the configured database URL."""

    url = get_db_url()
    return create_engine(url, echo=echo, future=True, **kwargs)


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Return a sessionmaker bound to the given engine (or the default)."""

    engine = engine or get_engine()
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


__all__ = ["get_engine", "get_session_factory"]
