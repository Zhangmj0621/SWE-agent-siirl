import ipaddress
import os
import socket

import psutil


def get_net_interface_ip():
    """
    Return (hostname, ip) based on GLOO_SOCKET_IFNAME env-var.
    IPv6 addresses are wrapped in brackets [].
    """
    ifname = os.getenv("GLOO_SOCKET_IFNAME")
    addrs = psutil.net_if_addrs()

    # pick interfaces to scan
    targets = (
        [(ifname, addrs[ifname])] if ifname in addrs else [(n, addrs[n]) for n in addrs if n not in ("lo", "Loopback Pseudo-Interface 1")]
    )

    # IPv4 first, then IPv6
    for family, prefix in ((socket.AF_INET, "127."), (socket.AF_INET6, "::1")):
        for _, snics in targets:
            for snic in snics:
                if snic.family == family and not snic.address.startswith(prefix):
                    ip = snic.address.split("%")[0]  # drop zone-index
                    if ipaddress.ip_address(ip).version == 6:
                        ip = f"[{ip}]"
                    return ip

    return "127.0.0.1"


def get_free_port(address: str) -> tuple[int, socket.socket]:
    """
    Get a free port and return (port, bound_socket).

    The socket remains bound until caller explicitly closes it,
    preventing port races between allocation and actual server binding.

    Usage:
        port, sock = get_free_port(address)
        # ... configure server with port ...
        sock.close()  # Close just before server starts
        server.start(port=port)

    Args:
        address: IP address to bind (IPv4 or IPv6)

    Returns:
        tuple[int, socket.socket]: (port, socket) - caller must close socket
    """
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((address, 0))
    return sock.getsockname()[1], sock
