"""
Entry point for the packaged desktop build.

PyInstaller freezes this file into `meet-companion-server`. The Electron
shell launches it with a port and a data directory, waits for /health, and
opens the window. Everything the app writes - database, config, model
weights - lives under that data directory, so the install itself stays
read-only and uninstalling is a matter of deleting one folder.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _bundle_root() -> Path:
    # PyInstaller unpacks data files next to the executable (onedir) or into
    # sys._MEIPASS (onefile); both are covered here.
    return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Meet Companion server (desktop build)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MEET_COMPANION_PORT", "0")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--data-dir", default=os.environ.get("MEET_COMPANION_DATA_DIR"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir or Path.home() / ".meet-companion").expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    # Relative defaults (data/…) resolve against the working directory, so
    # moving there is what points the whole app at the data directory.
    os.chdir(data_dir)
    os.environ.setdefault("MEET_COMPANION_CONFIG", str(data_dir / "data" / "config.json"))
    os.environ.setdefault("MEET_COMPANION_STATIC_DIR", str(_bundle_root() / "static"))
    # Embedding weights shipped with the build; copied into the data directory
    # on first start so search works without a download (see services/embedding).
    bundled_models = _bundle_root() / "models"
    if bundled_models.is_dir():
        os.environ.setdefault("MEET_COMPANION_BUNDLED_MODELS", str(bundled_models))
    # The tunnel ships with the build (scripts/fetch_cloudflared.py).
    for name in ("cloudflared.exe", "cloudflared"):
        bundled_tunnel = _bundle_root() / "bin" / name
        if bundled_tunnel.is_file():
            os.environ.setdefault("MEET_COMPANION_CLOUDFLARED", str(bundled_tunnel))
            break
    os.environ.setdefault("APP_ENV", "desktop")
    # No .env in a packaged install: one shipped by accident must not override
    # what the person configured in Settings.
    os.environ.setdefault("MEET_COMPANION_NO_DOTENV", "1")

    import uvicorn

    from app.main import app

    port = args.port or _free_port(args.host)
    # The automatic tunnel forwards to this port (app.services.tunnel).
    os.environ["MEET_COMPANION_PORT"] = str(port)
    # Printed before serving so the shell can read it from stdout.
    print(f"MEET_COMPANION_PORT={port}", flush=True)
    uvicorn.run(app, host=args.host, port=port, log_level="info")


def _free_port(host: str) -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


if __name__ == "__main__":
    main()
