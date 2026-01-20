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

import os
import socket

import numpy as np
import torch

from siirl.utils.backend.device import get_torch_device


def get_master_info() -> tuple[str, str]:
    """
    Get master address and port for distributed training.

    Priority:
    1. Use environment variables if already set (MASTER_ADDR, MASTER_PORT)
    2. Otherwise, get host IP and find a free port

    Note: If MASTER_ADDR is a hostname, it will be resolved to an IP address.

    Returns:
        A tuple containing (master_addr, master_port)
    """
    # Check if already set in environment
    master_addr = os.getenv("MASTER_ADDR")
    master_port = os.getenv("MASTER_PORT")

    if master_addr is None:
        # Get host IP address
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            master_addr = s.getsockname()[0]
            s.close()
        except Exception:
            master_addr = "127.0.0.1"
    else:
        # Resolve hostname to IP if necessary
        from loguru import logger

        try:
            socket.inet_aton(master_addr)
            # It's already a valid IP address
        except OSError:
            # It's a hostname, resolve it to IP
            try:
                resolved_ip = socket.gethostbyname(master_addr)
                logger.info(f"Resolved MASTER_ADDR '{master_addr}' -> '{resolved_ip}'")
                master_addr = resolved_ip
            except socket.gaierror as e:
                logger.warning(f"Failed to resolve hostname '{master_addr}': {e}")

    if master_port is None:
        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            master_port = str(s.getsockname()[1])

    return master_addr, master_port


def append_to_dict(data: dict, new_data: dict):
    """Append values from new_data to lists in data.

    For each key in new_data, this function appends the corresponding value to a list
    stored under the same key in data. If the key doesn't exist in data, a new list is created.

    Args:
        data: Target dictionary containing lists as values
        new_data: Source dictionary with values to append

    Example:
        >>> metrics = {}
        >>> append_to_dict(metrics, {"loss": 0.5})
        >>> append_to_dict(metrics, {"loss": 0.3})
        >>> metrics
        {'loss': [0.5, 0.3]}
    """
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


def set_random_seed(seed):
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)
