from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse


class PrivacyViolation(RuntimeError):
    pass


_ORIGINAL_CONNECT = None


def enforce_offline_model_runtime() -> None:
    """Disable model-hub telemetry and force model libraries into offline mode."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")


def _host_is_loopback(host: str) -> bool:
    normalized = host.strip().strip("[]").lower()
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def assert_local_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise PrivacyViolation(f"Only local HTTP(S) URLs are allowed, got: {url}")
    if not parsed.hostname or not _host_is_loopback(parsed.hostname):
        raise PrivacyViolation(
            f"Privacy Lock blocked a non-local endpoint: {url}. "
            "Only localhost / 127.0.0.1 / ::1 are permitted."
        )


def install_socket_guard() -> None:
    """Block outbound TCP connections except loopback.

    This is intentionally installed only in runtime processes. Explicit model download
    scripts do not install it.
    """
    global _ORIGINAL_CONNECT
    if _ORIGINAL_CONNECT is not None:
        return

    _ORIGINAL_CONNECT = socket.socket.connect

    def guarded_connect(sock: socket.socket, address):
        # AF_UNIX and other non-(host, port) address formats remain local.
        if isinstance(address, tuple) and address:
            host = str(address[0])
            if not _host_is_loopback(host):
                raise PrivacyViolation(
                    f"Privacy Lock blocked outbound socket connection to {host}."
                )
        return _ORIGINAL_CONNECT(sock, address)

    socket.socket.connect = guarded_connect


def activate_privacy_lock() -> None:
    enforce_offline_model_runtime()
    install_socket_guard()
