"""
The cloudflared the installer ships is pinned and checked: the right build
for each platform, and nothing whose checksum differs from Cloudflare's.
"""
import hashlib
import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_cloudflared as fc  # noqa: E402


@pytest.mark.parametrize("system,machine,asset", [
    ("Windows", "AMD64", "cloudflared-windows-amd64.exe"),
    ("Linux", "x86_64", "cloudflared-linux-amd64"),
    ("Linux", "aarch64", "cloudflared-linux-arm64"),
    ("Darwin", "arm64", "cloudflared-darwin-arm64.tgz"),
    ("Darwin", "x86_64", "cloudflared-darwin-amd64.tgz"),
])
def test_each_platform_gets_its_own_build(system, machine, asset):
    assert fc.asset_for(system, machine) == asset
    assert asset in fc.ASSETS  # and a pinned checksum for it


def test_a_download_that_does_not_match_the_published_checksum_is_refused():
    with pytest.raises(SystemExit, match="refusing to ship"):
        fc.verify("cloudflared-linux-amd64", b"not what cloudflare published")


def test_a_matching_download_passes(monkeypatch):
    data = b"pretend binary"
    monkeypatch.setitem(fc.ASSETS, "cloudflared-linux-amd64", hashlib.sha256(data).hexdigest())
    fc.verify("cloudflared-linux-amd64", data)


def test_the_macos_archive_is_unpacked_to_the_binary():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        payload = b"\xcf\xfa\xed\xfe mach-o"
        info = tarfile.TarInfo("cloudflared")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    assert fc.unpack("cloudflared-darwin-arm64.tgz", buffer.getvalue()) == b"\xcf\xfa\xed\xfe mach-o"
    assert fc.unpack("cloudflared-linux-amd64", b"elf") == b"elf"


def test_the_macos_binary_must_be_signed_by_cloudflare(monkeypatch, tmp_path):
    """A matching checksum is not enough on macOS: the signature must be Cloudflare's."""
    import subprocess as sp

    def fake_run(args, **kwargs):
        if "--verify" in args:
            return sp.CompletedProcess(args, 0, "", "")
        return sp.CompletedProcess(args, 0, "", chr(10).join(["Authority=Developer ID Application: Someone Else (ABC)", "Authority=Apple Root CA"]))

    monkeypatch.setattr(fc.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="not signed by"):
        fc.verify_macos_signature(tmp_path / "cloudflared")

    def cloudflare_run(args, **kwargs):
        if "--verify" in args:
            return sp.CompletedProcess(args, 0, "", "")
        return sp.CompletedProcess(args, 0, "", chr(10).join(["Authority=Developer ID Application: Cloudflare Inc (68WVV388M8)", "Authority=Apple Root CA"]))

    monkeypatch.setattr(fc.subprocess, "run", cloudflare_run)
    fc.verify_macos_signature(tmp_path / "cloudflared")


def test_a_broken_signature_is_refused(monkeypatch, tmp_path):
    import subprocess as sp

    monkeypatch.setattr(fc.subprocess, "run", lambda args, **kw: sp.CompletedProcess(args, 1, "", "invalid signature (code or signature have been modified)"))
    with pytest.raises(SystemExit, match="fails codesign"):
        fc.verify_macos_signature(tmp_path / "cloudflared")
