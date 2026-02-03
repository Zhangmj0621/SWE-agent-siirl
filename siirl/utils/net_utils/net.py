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


def is_port_available(port: int, address: str = "") -> bool:
    """
    Check if a port is available by attempting to bind to it.

    Uses SO_REUSEADDR only (not SO_REUSEPORT) for accurate availability check.
    This matches slime's implementation.

    Args:
        port: Port number to check
        address: IP address to bind (empty for all interfaces)

    Returns:
        True if port is available
    """
    family, addr = _get_socket_family(address)
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((addr, port))
            sock.listen(1)
            return True
    except OSError:
        return False


def get_free_port(address: str = "", start_port: int = DEFAULT_START_PORT) -> int:
    """
    Find a free port starting from start_port using sequential allocation (slime-style).

    Args:
        address: IP address to bind (empty for all interfaces)
        start_port: Starting port number (default: 15000)

    Returns:
        Available port number

    Raises:
        RuntimeError: If no available port found
    """
    for port in range(start_port, MAX_PORT):
        if is_port_available(port, address):
            return port
    raise RuntimeError(f"No available port in range [{start_port}, {MAX_PORT})")
