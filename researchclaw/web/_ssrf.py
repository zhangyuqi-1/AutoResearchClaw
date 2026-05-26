"""SSRF validation for URLs fetched by the web layer."""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse


_SUSPICIOUS_URL_RE = re.compile(r"[\\@]")


def check_url_ssrf(url: str) -> str | None:
    """Return an error message if *url* targets a private/internal host.

    Validates scheme (http/https only) and resolves the hostname to check
    against all RFC 1918, loopback, link-local, and reserved IP ranges
    using :func:`ipaddress.ip_address`.

    Returns ``None`` if the URL is safe to fetch.
    """
    if "\\" in url:
        return "Blocked URL containing backslash (potential SSRF bypass)"
    if "@" in url.split("//", 1)[-1].split("/", 1)[0]:
        return "Blocked URL containing userinfo (potential SSRF bypass)"
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Unsupported URL scheme: {parsed.scheme}"
    hostname = parsed.hostname or ""
    if not hostname:
        return "URL has no hostname"
    # Try parsing hostname as a literal IP address first
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        # It's a domain name — resolve to IP
        try:
            info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            addr = ipaddress.ip_address(info[0][4][0])
        except (socket.gaierror, OSError, IndexError):
            # Can't resolve — let the actual request fail naturally
            return None
    if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
        return f"Blocked internal/private URL: {hostname}"
    return None
