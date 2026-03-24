"""Network utilities."""

from __future__ import annotations

import socket


def is_port_open(port: int, host: str = "localhost", timeout: float = 2) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
