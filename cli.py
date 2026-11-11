"""Compatibility launcher: ``python cli.py`` → ``fourx.cli.main``."""

from fourx.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
