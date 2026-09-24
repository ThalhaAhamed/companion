# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the Meet Companion server binary.

Build from the repository root:

    pyinstaller desktop/server.spec

Produces dist/meet-companion-server/ (one directory - starts faster than a
single-file build and is what electron-builder copies into the app). The
built web UI is expected in frontend/dist and is shipped under static/.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve().parent

hiddenimports = []
datas = [
    (str(ROOT / "frontend" / "dist"), "static"),
    (str(ROOT / "alembic.ini"), "."),
    (str(ROOT / "app" / "migrations"), "app/migrations"),
]
binaries = []

# Embedding model weights, pre-fetched by scripts/fetch_embedding_model.py so
# a fresh install can search without downloading anything. Optional: a build
# without them still works, it just fetches on first use like a server does.
MODELS = ROOT / "desktop" / "models"
if MODELS.is_dir() and any(MODELS.iterdir()):
    datas.append((str(MODELS), "models"))

# cloudflared, for the automatic tunnel, fetched and checksum-verified by
# scripts/fetch_cloudflared.py. A binary, not data, so it keeps its
# executable bit on macOS and Linux. Optional: without it the Settings
# switch says the tunnel is unavailable.
TUNNEL_BIN = ROOT / "desktop" / "bin"
for name in ("cloudflared.exe", "cloudflared"):
    if (TUNNEL_BIN / name).is_file():
        binaries.append((str(TUNNEL_BIN / name), "bin"))

# fastembed/onnxruntime/tokenizers ship native libraries and data files that
# static analysis does not find on its own.
for package in ("fastembed", "onnxruntime", "tokenizers"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

# uvicorn's workers, and the async DB drivers, are imported by name.
hiddenimports += collect_submodules("uvicorn")
hiddenimports += ["aiosqlite", "asyncpg", "pgvector", "pgvector.sqlalchemy", "app.desktop_entry"]
hiddenimports += collect_submodules("app")
# Alembic drives schema migrations at first run inside the frozen server.
hiddenimports += collect_submodules("alembic")

a = Analysis(
    [str(ROOT / "app" / "desktop_entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Never needed at runtime; keeps the bundle small.
        "torch", "sentence_transformers", "transformers", "tkinter", "pytest",
        "matplotlib", "IPython", "notebook", "psycopg2",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="meet-companion-server",
    debug=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="meet-companion-server",
)
