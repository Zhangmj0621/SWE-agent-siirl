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
    Ask the OS for a free port (ephemeral range) and return
    (port, bound_socket). The socket is reused by caller.
    """
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((address, 0))
        return s.getsockname()[1]
