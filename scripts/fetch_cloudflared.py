"""
Fetch the cloudflared binary the desktop build ships, for the automatic
tunnel (app/services/tunnel.py): a laptop gets a public address for
MeetStream without anyone installing or running anything.

Pinned to one release, and every download is checked against the SHA-256
Cloudflare published for it - a build never ships a binary that is merely
"whatever the latest URL served today". The release workflow runs this
before PyInstaller; desktop/server.spec ships desktop/bin as bin/, and
app/desktop_entry.py points the server at it.

    python scripts/fetch_cloudflared.py            # into desktop/bin
    python scripts/fetch_cloudflared.py some/dir
"""
from __future__ import annotations

import hashlib
import io
import platform
import stat
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

VERSION = "2026.9.1"

#: Asset name -> SHA-256 of the file on the 2026.9.1 release.
#:
#: Windows and Linux: the "SHA256 Checksums" in the release notes, which
#: also match GitHub's own digest of each uploaded asset.
#:
#: macOS: GitHub's recorded digest of the uploaded asset. The release notes
#: list different hashes for the two .tgz files than the files Cloudflare
#: actually published (uploaded 2026-09-11, never changed since) - found when
#: the v0.6.5 build refused the download. So these are pinned to what was
#: uploaded, and on macOS the binary must also carry Cloudflare's Developer
#: ID signature (verify_macos_signature), which a substituted file cannot.
ASSETS = {
    "cloudflared-windows-amd64.exe": "2837888cc0f5d58f15b6dc478376de90b4d3ba5241c7947455d1e0a0df429712",
    "cloudflared-linux-amd64": "03f1f25d1cc93b9ad6c60569d44060bc4f17ed97075760ed8cfca4b12dcd68cc",
    "cloudflared-linux-arm64": "3d97437c71848bd8df68041e12436b484a661d95073ea1937f01a845ce88faa3",
    "cloudflared-darwin-amd64.tgz": "ff0d3b51d5ff70eceef89d6b32145fee985018a2174596a5dbe405e2766e2ac4",
    "cloudflared-darwin-arm64.tgz": "c27ab8fd0aa489449e3d201eb02f957ef460a13b613662928b1b23394bf1bcfe",
}

#: Who must have signed the macOS binary.
MACOS_SIGNER = "Developer ID Application: Cloudflare"


def asset_for(system: str, machine: str) -> str:
    """The release asset for this platform - the same one the frozen server targets."""
    arch = "arm64" if machine.lower() in ("arm64", "aarch64") else "amd64"
    if system == "Windows":
        return "cloudflared-windows-amd64.exe"
    if system == "Darwin":
        return f"cloudflared-darwin-{arch}.tgz"
    if system == "Linux":
        return f"cloudflared-linux-{arch}"
    raise SystemExit(f"No cloudflared build for {system}/{machine}")


def fetch(asset: str) -> bytes:
    url = f"https://github.com/cloudflare/cloudflared/releases/download/{VERSION}/{asset}"
    last = None
    for _ in range(3):  # the release CDN has failed builds before on one 5xx
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last = exc
    raise SystemExit(f"Could not download {url}: {last}")


def verify(asset: str, data: bytes) -> None:
    digest = hashlib.sha256(data).hexdigest()
    expected = ASSETS[asset]
    if digest != expected:
        raise SystemExit(f"{asset}: SHA-256 {digest} does not match the published {expected}; refusing to ship it.")


def unpack(asset: str, data: bytes) -> bytes:
    """The macOS builds come as a .tgz holding a single `cloudflared`."""
    if not asset.endswith(".tgz"):
        return data
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        member = next((m for m in archive.getmembers() if Path(m.name).name == "cloudflared" and m.isfile()), None)
        if member is None:
            raise SystemExit(f"{asset} has no cloudflared inside")
        return archive.extractfile(member).read()


def verify_macos_signature(path: Path) -> None:
    """The binary is signed by Cloudflare's Apple Developer ID, and the signature is intact."""
    check = subprocess.run(["codesign", "--verify", "--strict", str(path)], capture_output=True, text=True)
    if check.returncode != 0:
        raise SystemExit(f"{path} fails codesign verification: {check.stderr.strip()}; refusing to ship it.")
    details = subprocess.run(["codesign", "-dv", "--verbose=2", str(path)], capture_output=True, text=True)
    authorities = [line for line in details.stderr.splitlines() if line.startswith("Authority=")]
    if not any(MACOS_SIGNER in line for line in authorities):
        raise SystemExit(f"{path} is not signed by {MACOS_SIGNER} ({authorities or 'unsigned'}); refusing to ship it.")


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else PROJECT_ROOT / "desktop" / "bin").resolve()
    asset = asset_for(platform.system(), platform.machine())
    data = fetch(asset)
    verify(asset, data)
    binary = unpack(asset, data)

    target.mkdir(parents=True, exist_ok=True)
    out = target / ("cloudflared.exe" if asset.endswith(".exe") else "cloudflared")
    out.write_bytes(binary)
    out.chmod(out.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    if platform.system() == "Darwin":
        verify_macos_signature(out)

    version = subprocess.run([str(out), "--version"], capture_output=True, text=True, timeout=30)
    if VERSION not in (version.stdout + version.stderr):
        raise SystemExit(f"{out} does not report {VERSION}: {version.stdout or version.stderr}")
    signed = ", Cloudflare-signed" if platform.system() == "Darwin" else ""
    print(f"cloudflared {VERSION} ({asset}, sha256 verified{signed}) ready at {out}: {out.stat().st_size / 1_048_576:.0f} MB")


if __name__ == "__main__":
    main()
