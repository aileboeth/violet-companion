"""Signed releases (Violet 0.7.59; user 2026-11-08: updates through GitHub mirrors must not be able to run code
the mirror changed). Ed25519 (RFC 8032) in the standard library only -- the installer and the root update service have
no third-party packages.

  * Violet releases: GitHub Actions signs release.json with the secret VIOLET_RELEASE_SIGNING_KEY (release.yml ->
    make_release.py --sign-key-env); release.json carries the SHA-256 of install.sh and violet.tar.gz. The root update
    service checks the signature with VIOLET_RELEASE_KEY before it runs anything it downloaded through a mirror.
  * Violet MCP relays: tools/package_release.py of Violet MCP signs SHA256SUMS.txt (SHA256SUMS.txt.sig) with its own
    key; `violet mcp-relay install` checks it with MCP_RELAY_KEY when the files come through a mirror.

Signature files hold the 64-byte signature in base64 on one line. A key file / secret is the 32-byte seed as 64 hex
characters. The private keys never live in this repository (.gitignore: *.key; local/ is ignored).

    python tools/violet_sign.py keygen <key file>                  new key; prints only the public key
    python tools/violet_sign.py sign --key-env NAME <file>         writes <file>.sig
    python tools/violet_sign.py verify --public <base64> <file>    checks <file>.sig

The arithmetic follows RFC 8032 section 6 (not constant-time: signing only runs on the release machine / CI).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
from pathlib import Path

# the public halves (generated 2026-11-08; the private halves: GitHub secret VIOLET_RELEASE_SIGNING_KEY and
# Violet MCP's signing/ folder)
VIOLET_RELEASE_KEY = "yZ7x+Dur6n3PEdkUhzsJKctMwhWZ5e9ZjJEYMAVWOeg="
MCP_RELAY_KEY = "Tf0YPcokxMSZwmpDfqucHE8HZqpgv0NXXcY+d78nw0I="

_P = 2 ** 255 - 19
_Q = 2 ** 252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _inv(x: int) -> int:
    return pow(x, _P - 2, _P)


def _add(a: tuple, b: tuple) -> tuple:
    x1 = (a[1] - a[0]) * (b[1] - b[0]) % _P
    x2 = (a[1] + a[0]) * (b[1] + b[0]) % _P
    c = 2 * a[3] * b[3] * _D % _P
    d = 2 * a[2] * b[2] % _P
    e, f, g, h = x2 - x1, d - c, d + c, x2 + x1
    return e * f % _P, g * h % _P, f * g % _P, e * h % _P


def _mul(s: int, point: tuple) -> tuple:
    out = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            out = _add(out, point)
        point = _add(point, point)
        s >>= 1
    return out


def _equal(a: tuple, b: tuple) -> bool:
    return (a[0] * b[2] - b[0] * a[2]) % _P == 0 and (a[1] * b[2] - b[1] * a[2]) % _P == 0


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    x2 = (y * y - 1) * _inv(_D * y * y + 1)
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_GY = 4 * _inv(5) % _P
_GX = _recover_x(_GY, 0)
_G = (_GX, _GY, 1, _GX * _GY % _P)


def _compress(point: tuple) -> bytes:
    zinv = _inv(point[2])
    x, y = point[0] * zinv % _P, point[1] * zinv % _P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decompress(data: bytes) -> tuple | None:
    if len(data) != 32:
        return None
    y = int.from_bytes(data, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


def _expand(seed: bytes) -> tuple[int, bytes]:
    if len(seed) != 32:
        raise ValueError("an Ed25519 key is 32 bytes")
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def _h(data: bytes) -> int:
    return int.from_bytes(hashlib.sha512(data).digest(), "little")


def public_key(seed: bytes) -> bytes:
    return _compress(_mul(_expand(seed)[0], _G))


def sign(seed: bytes, message: bytes) -> bytes:
    a, prefix = _expand(seed)
    pub = _compress(_mul(a, _G))
    r = _h(prefix + message) % _Q
    big_r = _compress(_mul(r, _G))
    s = (r + _h(big_r + pub + message) % _Q * a) % _Q
    return big_r + int.to_bytes(s, 32, "little")


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    if len(public) != 32 or len(signature) != 64:
        return False
    point_a = _decompress(public)
    point_r = _decompress(signature[:32])
    if point_a is None or point_r is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _Q:
        return False
    h = _h(signature[:32] + public + message) % _Q
    return _equal(_mul(s, _G), _add(point_r, _mul(h, point_a)))


def signature_text(seed: bytes, message: bytes) -> str:
    """The content of a .sig file."""
    return base64.b64encode(sign(seed, message)).decode("ascii") + "\n"


def check(public_b64: str, message: bytes, sig_text: bytes | str | None) -> bool:
    """True only for a valid signature of ``message`` by the key ``public_b64`` (a .sig file's content)."""
    if not sig_text or public_b64.startswith("__"):
        return False
    try:
        public = base64.b64decode(public_b64, validate=True)
        raw = sig_text.encode("ascii") if isinstance(sig_text, str) else bytes(sig_text)
        signature = base64.b64decode(raw.strip(), validate=True)
    except (ValueError, UnicodeError):
        return False
    return verify(public, message, signature)


def seed_from_text(text: str) -> bytes:
    value = (text or "").strip()
    try:
        seed = bytes.fromhex(value)
    except ValueError:
        raise SystemExit("the signing key must be 64 hex characters") from None
    if len(seed) != 32:
        raise SystemExit("the signing key must be 64 hex characters")
    return seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="violet_sign")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("keygen")
    gen.add_argument("key_file", type=Path)
    sig = sub.add_parser("sign")
    sig.add_argument("--key-env")
    sig.add_argument("--key-file", type=Path)
    sig.add_argument("file", type=Path)
    ver = sub.add_parser("verify")
    ver.add_argument("--public", required=True)
    ver.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    if args.command == "keygen":
        if args.key_file.exists():
            raise SystemExit(f"{args.key_file} already exists; not overwritten")
        seed = os.urandom(32)
        args.key_file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(args.key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="ascii") as fh:
            fh.write(seed.hex() + "\n")
        print(base64.b64encode(public_key(seed)).decode("ascii"))       # the public key only
        return 0
    if args.command == "sign":
        if args.key_env:
            text = os.environ.get(args.key_env, "")
            if not text.strip():
                raise SystemExit(f"the signing key {args.key_env} is not set")
        elif args.key_file:
            text = args.key_file.read_text(encoding="ascii")
        else:
            raise SystemExit("--key-env or --key-file")
        target = args.file.with_name(args.file.name + ".sig")
        target.write_text(signature_text(seed_from_text(text), args.file.read_bytes()), encoding="ascii")
        print(target)
        return 0
    sig_path = args.file.with_name(args.file.name + ".sig")
    ok = check(args.public, args.file.read_bytes(), sig_path.read_bytes() if sig_path.is_file() else None)
    print("OK" if ok else "BAD")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
