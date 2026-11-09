"""The HTTPS server of the Control Center (Violet stage 3).

werkzeug's own HTTPS wraps the LISTENING socket: every TLS handshake then runs inside accept(), in the one thread that
accepts all connections. A single client that connects and never finishes its handshake -- a browser's speculative
pre-connection, a connection left behind by a certificate warning, a mobile network that drops mid-way -- blocks
every other connection until it goes away. The login page (one request, styles inline) still showed, but the Control
Center's stylesheet and scripts (ten more requests) timed out: the page was plain, unstyled text.

Here the listening socket stays plain; each accepted connection does its handshake in its own worker thread, with a
time limit, so a slow or idle client only ever holds up itself.
"""

from __future__ import annotations

import re
import socket
import ssl
from typing import Any

from werkzeug.serving import ThreadedWSGIServer

HANDSHAKE_SECONDS = 10.0
IDLE_SECONDS = 120.0          # a kept-alive connection with nothing to say is closed after this


class ThreadedTLSServer(ThreadedWSGIServer):
    def __init__(self, host: str, port: int, app: Any, ssl_context: ssl.SSLContext, **kwargs: Any):
        super().__init__(host, port, app, ssl_context=None, **kwargs)   # plain listening socket
        self.ssl_context = ssl_context        # werkzeug reads this: wsgi.url_scheme = "https", SSL error handling

    def finish_request(self, request: socket.socket, client_address: Any) -> None:  # type: ignore[override]
        """Runs in the connection's own thread (ThreadingMixIn)."""
        request.settimeout(HANDSHAKE_SECONDS)
        try:
            first = request.recv(1, socket.MSG_PEEK)
        except OSError:
            return
        if first and first != b"\x16":          # not a TLS handshake record: someone typed http:// -- send them to https
            _redirect_to_https(request, self.server_address[1])
            return
        try:
            tls = self.ssl_context.wrap_socket(request, server_side=True)       # the handshake, here
        except (ssl.SSLError, OSError):
            return                            # not TLS, too slow, gone: only this connection is dropped
        tls.settimeout(IDLE_SECONDS)
        try:
            super().finish_request(tls, client_address)
        finally:
            try:
                tls.close()
            except OSError:
                pass


_HOST_RE = re.compile(r"^(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9.-]{1,253})(:\d{1,5})?$")
_PATH_RE = re.compile(r"^/[A-Za-z0-9._~!$&'()*+,;=:@%/?-]{0,2000}$")


def _redirect_to_https(conn: socket.socket, port: int) -> None:
    """2026-11-09: a plain-HTTP request on the HTTPS port used to get a dropped connection ("empty response" in the
    browser). Now it gets 301 to the same host and path over https. Nothing is read beyond the request head and no
    request reaches the app; a head that cannot be understood gets a plain 400."""
    data = b""
    try:
        while b"\r\n\r\n" not in data and len(data) < 8192:
            chunk = conn.recv(2048)
            if not chunk:
                break
            data += chunk
    except OSError:
        pass
    lines = data.split(b"\r\n")
    parts = lines[0].split(b" ") if lines else []
    path = parts[1].decode("latin-1") if len(parts) == 3 and parts[2].startswith(b"HTTP/") else ""
    host = ""
    for line in lines[1:]:
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"host":
            host = value.strip().decode("latin-1")
            break
    if host and _HOST_RE.match(host) and _PATH_RE.match(path):
        if ":" not in host.rsplit("]", 1)[-1]:              # no port given (a proxy in front): keep ours
            host = f"{host}:{port}"
        location = f"https://{host}{path}"
        body = b"Use https://\n"
        head = (f"HTTP/1.1 301 Moved Permanently\r\nLocation: {location}\r\nContent-Type: text/plain\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n")
    else:
        body = b"This port speaks HTTPS only.\n"
        head = f"HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    try:
        conn.sendall(head.encode("latin-1") + body)
    except OSError:
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def make_tls_server(host: str, port: int, app: Any, ssl_context: ssl.SSLContext) -> ThreadedTLSServer:
    return ThreadedTLSServer(host, port, app, ssl_context)
