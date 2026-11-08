"""Build the files of a Violet release (phase 10.7) -- what the one-command install downloads.

    python tools/make_release.py --base https://example.org/violet/latest --out dist/          (from git HEAD)
    python tools/make_release.py --base http://127.0.0.1:8000 --out <dir> --worktree        (tests: working tree)

Output (upload all of it to the address given as --base):
  install.sh                   the installer with the download address filled in
                               -> users run:  curl -fsSL <base>/install.sh | bash   (root or a sudo user)
  violet.tar.gz           the program (folder violet/): tracked files of "violet" at HEAD, without
                               tests, local data, .env, databases or reports
  violet.tar.gz.sha256    its SHA-256 (install.sh refuses an archive that does not match)
  SHA256SUMS, release.json     checksums of everything; version, commit, build time, the SHA-256 of install.sh and
                               the archive, the release commit's first line
  release.json.sig             with --sign-key-env (0.7.59): its Ed25519 signature (tools/violet_sign.py) -- updates
                               through a GitHub mirror need it

No secret is ever packaged: the archive is checked for .env, *.db, credentials, keys and signing material, and the
build fails if one is found.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import fnmatch
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
REPO = PROJECT.parent
PLACEHOLDER = "__VIOLET_RELEASE_BASE__"
EXCLUDE = ("tests/*", "*/__pycache__/*", "__pycache__/*", "*.pyc", "tools/output/*", "data/*", "logs/*", ".venv*",
           "*.bak", "main.pid", ".pytest_cache/*")
FORBIDDEN = (".env", ".env.*", "*.db", "*.db-wal", "*.db-shm", "credentials.json", "*.key", "*.jks", "*.keystore",
             "*.p12", "*.pem", "admin_token", "session_secret", "secrets.json", "channel_secrets.json",
             "mcp_secrets.json", "device.secret")


def _excluded(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, pattern) for pattern in EXCLUDE)


def _forbidden(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(name, pattern) for pattern in FORBIDDEN) and not name.endswith(("env_example",))


def files_from_git(ref: str = "HEAD") -> list[tuple[str, bytes, int]]:
    """(path inside the program folder, content, mode) of every tracked file of "violet" at ref."""
    listing = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "-z", ref, "--", "violet/"],
                             capture_output=True, check=True).stdout
    out = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        mode, kind, sha = meta.decode().split()
        if kind != "blob":
            continue
        rel = path.decode("utf-8")[len("violet/"):]
        if _excluded(rel):
            continue
        content = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", sha], capture_output=True,
                                 check=True).stdout
        out.append((rel, content, 0o755 if mode == "100755" or rel.endswith(".sh") else 0o644))
    return out


def files_from_worktree() -> list[tuple[str, bytes, int]]:
    out = []
    for path in sorted(PROJECT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(PROJECT).as_posix()
        if _excluded(rel) or _forbidden(rel) or rel.startswith(("data/", "logs/")) or "/.pytest_cache/" in rel:
            continue
        out.append((rel, path.read_bytes(), 0o755 if rel.endswith(".sh") else 0o644))
    return out


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _subject() -> str:
    """The release commit's first line (the update dialog shows it as the release notes)."""
    try:
        return subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%s"], capture_output=True, text=True,
                              encoding="utf-8", check=True).stdout.strip()[:2000]
    except (OSError, subprocess.CalledProcessError):
        return ""


def build(base: str, out: Path, worktree: bool = False, ref: str = "HEAD", sign_key: str | None = None) -> dict:
    if not base.startswith(("https://", "http://127.0.0.1", "http://localhost")):
        raise SystemExit("--base must be an https:// address (http only for 127.0.0.1 tests)")
    base = base.rstrip("/")
    files = files_from_worktree() if worktree else files_from_git(ref)
    bad = [rel for rel, _c, _m in files if _forbidden(rel)]
    if bad:
        raise SystemExit("refusing to package secrets / data: " + ", ".join(bad))
    if not any(rel == "tools/violet_installer.py" for rel, _c, _m in files):
        raise SystemExit("the installer is missing from the program files")
    version = next((c.decode().strip() for rel, c, _m in files if rel == "VERSION"), "unknown")
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime(2026, 1, 1).timestamp() if not worktree else datetime.datetime.now().timestamp()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
        for rel, content, mode in sorted(files):
            info = tarfile.TarInfo("violet/" + rel)
            info.size, info.mode, info.mtime = len(content), mode, int(stamp)
            tar.addfile(info, io.BytesIO(content.replace(b"\r\n", b"\n") if rel.endswith(".sh") else content))
    archive = out / "violet.tar.gz"
    archive.write_bytes(buffer.getvalue())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / "violet.tar.gz.sha256").write_text(digest + "  violet.tar.gz\n", encoding="utf-8")
    installer = next(c for rel, c, _m in files if rel == "install.sh").decode("utf-8").replace("\r\n", "\n")
    (out / "install.sh").write_bytes(installer.replace(PLACEHOLDER, base).encode("utf-8"))
    meta = {"version": version, "commit": "" if worktree else _commit(), "worktree": worktree, "base": base,
            "built_at": datetime.datetime.now().isoformat(timespec="seconds"), "files": len(files),
            "install_command": f"curl -fsSL {base}/install.sh | bash",
            # 0.7.59: what the root update service checks before running anything (through the signature below)
            "sha256": {"install.sh": hashlib.sha256((out / "install.sh").read_bytes()).hexdigest(),
                       "violet.tar.gz": digest},
            "notes": "" if worktree else _subject()}
    body = json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8")
    (out / "release.json").write_bytes(body)
    names = ["install.sh", "violet.tar.gz", "violet.tar.gz.sha256", "release.json"]
    if sign_key is not None:
        try:
            import violet_sign                          # run as a script: tools/ is on sys.path
        except ImportError:
            from tools import violet_sign
        sig =violet_sign.signature_text(violet_sign.seed_from_text(sign_key), body)
        public = base64.b64encode(violet_sign.public_key(violet_sign.seed_from_text(sign_key))).decode("ascii")
        if public != violet_sign.VIOLET_RELEASE_KEY and not worktree:
            raise SystemExit("the signing key is not the one whose public half is in tools/violet_sign.py")
        (out / "release.json.sig").write_text(sig, encoding="ascii")
        names.append("release.json.sig")
    sums = [f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}" for n in names]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_release")
    parser.add_argument("--base", required=True, help="where the release files will be served from")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worktree", action="store_true", help="package the working tree (tests) instead of git")
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--sign-key-env", help="sign release.json with the Ed25519 seed (hex) in this variable "
                                               "(release.yml: the secret VIOLET_RELEASE_SIGNING_KEY)")
    args = parser.parse_args(argv)
    key = None
    if args.sign_key_env:
        key = os.environ.get(args.sign_key_env, "").strip()
        if not key:
            raise SystemExit(f"{args.sign_key_env} is empty: add the repository secret before publishing a release "
                             "(Settings -> Secrets and variables -> Actions)")
    meta = build(args.base, args.out, worktree=args.worktree, ref=args.ref, sign_key=key)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
