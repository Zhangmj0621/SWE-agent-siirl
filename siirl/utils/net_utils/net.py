# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ipaddress
import os
import socket

import psutil

# Port allocation ranges (avoid ephemeral 32768-65536 and Ray 10002-19999)
DEFAULT_START_PORT = 15000
MAX_PORT = 32000


def get_net_interface_ip() -> str:
    """Get network interface IP based on GLOO_SOCKET_IFNAME env-var. IPv6 wrapped in []."""
    ifname = os.getenv("GLOO_SOCKET_IFNAME")
    addrs = psutil.net_if_addrs()

    targets = (
        [(ifname, addrs[ifname])] if ifname in addrs else [(n, addrs[n]) for n in addrs if n not in ("lo", "Loopback Pseudo-Interface 1")]
    )

    for family, prefix in ((socket.AF_INET, "127."), (socket.AF_INET6, "::1")):
        for _, snics in targets:
            for snic in snics:
                if snic.family == family and not snic.address.startswith(prefix):
                    ip = snic.address.split("%")[0]
                    return f"[{ip}]" if ipaddress.ip_address(ip).version == 6 else ip

    return "127.0.0.1"


def _get_socket_family(address: str) -> tuple[int, str]:
    """Return (socket_family, cleaned_address) for the given address."""
    if address and ":" in address.strip("[]"):
        return socket.AF_INET6, address.strip("[]")
    return socket.AF_INET, address


def _try_bind_port(port: int, address: str) -> socket.socket | None:
    """
    Try to bind to a port and return the socket if successful.

    Uses SO_REUSEADDR and SO_REUSEPORT for compatibility.
    The socket remains bound - caller is responsible for closing it.
    """
    family, addr = _get_socket_family(address)
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind((addr, port))
        sock.listen(1)
        return sock
    except OSError:
        return None


def is_port_available(port: int, address: str = "", strict: bool = False) -> bool:
    """
    Check if a port is available by attempting to bind to it.

    Args:
        port: Port number to check
        address: IP address to bind (empty for all interfaces)
        strict: If True, don't use SO_REUSEPORT during availability checks

    Returns:
        True if port is available
    """
    if strict:
        # Strict mode: don't use SO_REUSEPORT
        family, addr = _get_socket_family(address)
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((addr, port))
                sock.listen(1)
                return True
        except OSError:
            return False
    else:
        sock = _try_bind_port(port, address)
        if sock:
            sock.close()
            return True
        return False


def get_free_port(address: str = "", start_port: int = DEFAULT_START_PORT, strict: bool = False) -> int:
    """
    Find a free port starting from start_port using sequential allocation.

    Args:
        address: IP address to bind (empty for all interfaces)
        start_port: Starting port number (default: 15000)
        strict: If True, don't use SO_REUSEPORT during availability checks

    Returns:
        Available port number

    Raises:
        RuntimeError: If no available port found
    """
    for port in range(start_port, MAX_PORT):
        if is_port_available(port, address, strict=strict):
            return port
    raise RuntimeError(f"No available port in range [{start_port}, {MAX_PORT})")


def get_free_port_with_socket(address: str = "", start_port: int = DEFAULT_START_PORT) -> tuple[int, socket.socket]:
    """
    Find a free port and return (port, bound_socket) with socket held open.

    Combines slime's sequential allocation with verl's socket holding strategy:
    - Sequential port search from start_port (avoids conflicts)
    - Socket remains bound until caller closes it (minimizes race window)
    - Uses SO_REUSEPORT for compatibility with SGLang binding

    Args:
        address: IP address to bind (empty for all interfaces)
        start_port: Starting port number (default: 15000)

    Returns:
        tuple[int, socket.socket]: (port, socket) - caller must close socket

    Raises:
        RuntimeError: If no available port found
    """
    for port in range(start_port, MAX_PORT):
        sock = _try_bind_port(port, address)
        if sock:
            return port, sock
    raise RuntimeError(f"No available port in range [{start_port}, {MAX_PORT})")
