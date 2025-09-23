"""Module entry point to invoke the project CLI."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover - simple passthrough
    raise SystemExit(main())
