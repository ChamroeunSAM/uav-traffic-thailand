"""Logging setup with the Windows UTF-8 stdout fix baked in."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO, *, utf8: bool = True) -> None:
    """Configure root logger with a single stream handler.

    On Windows, sys.stdout/stderr default to cp1252 and crash on emoji or
    non-Latin glyphs that ultralytics/rfdetr both emit. We reconfigure to
    UTF-8 first (safe to call from any script entry point).
    """
    if utf8 and sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            except Exception:
                pass

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
