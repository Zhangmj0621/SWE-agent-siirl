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
import asyncio
import logging
import multiprocessing
import threading
import time
from typing import Any, Literal

import httpx
import requests
from loguru import logger
from requests.exceptions import RequestException

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
_thread_local = threading.local()


def wait_until_ok(
    url: str,
    *,
    process: "multiprocessing.Process" = None,
    max_wait: int = 3000,
    interval: int = 2,
    timeout: int = 3000,
    extra_headers: dict | None = None,
) -> None:
    """Block the execution until the given URL returns HTTP 200 status code or reaches max wait time.

    Args:
        url: Target URL to check health status
        process: Optional multiprocessing.Process object to monitor server status
        max_wait: Maximum wait time in seconds before raising exception
        interval: Interval in seconds between consecutive health checks
        timeout: Request timeout in seconds for each health check
        extra_headers: Optional additional headers for the health check request

    Raises:
        RuntimeError: If server process terminates unexpectedly or health check times out
    """
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        # Check if server process is still alive
        if process and not process.is_alive():
            raise RuntimeError(f"Server process terminated unexpectedly. Process: {process}, Alive status: {process.is_alive()}")
        try:
            # Send health check request
            if requests.get(url, timeout=timeout, headers=extra_headers or {}).status_code == 200:
                return
        except RequestException as exc:
            logger.debug("Health check request failed: %s", exc)
        # Wait for next check
        time.sleep(interval)

    raise RuntimeError(f"Health check failed after {max_wait} seconds for URL: {url}")


# Constant definitions (can be extracted to config file)
DEFAULT_TIMEOUT = 30.0  # Default timeout in seconds
DEFAULT_MAX_ATTEMPTS = 3  # Default maximum retry attempts
DEFAULT_RETRY_DELAY = 1.0  # Default initial retry delay in seconds
HTTPMethod = Literal["GET", "POST", "PUT", "DELETE"]  # Restrict supported HTTP methods


class GlobalAsyncHTTPClient:
    """
    A thread-safe, asynchronous HTTP client that creates an isolated client instance for each thread.
    This prevents issues with event loop mismatches when used across multiple threads or Ray actors.
    """

    _connect_timeout: float = 10.0  # Connection timeout in seconds

    @classmethod
    async def _get_client(cls) -> httpx.AsyncClient:
        """
        Retrieves or initializes the thread-local AsyncClient instance.

        Returns:
            httpx.AsyncClient: The thread-local HTTP client instance.
        """
        if not hasattr(_thread_local, "client"):
            _thread_local.client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=None),  # Unlimited connections (adjust based on your needs)
                timeout=httpx.Timeout(
                    connect=cls._connect_timeout,
                    read=None,  # Disable read timeout for long-running operations (e.g., model generation)
                    write=None,
                    pool=None,
                ),
                http2=False,  # Disable HTTP/2 for wider compatibility
                follow_redirects=True,  # Automatically follow HTTP redirects
            )
        return _thread_local.client

    @classmethod
    async def make_request(
        cls,
        url: str,
        payload: dict[str, Any] | None = None,
        method: HTTPMethod = "POST",
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int | None = None,
        retry_delay: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Makes an asynchronous HTTP request with an exponential backoff retry mechanism.

        Args:
            url: The target URL for the request.
            payload: Optional JSON payload to send with the request.
            method: The HTTP method to use (GET, POST, PUT, DELETE).
            timeout: The request timeout in seconds (only affects the connect timeout).
            max_attempts: The maximum number of retry attempts (overrides default).
            retry_delay: The initial delay between retries in seconds (overrides default).

        Returns:
            Optional[Dict[str, Any]]: The JSON response parsed as a dictionary if the request succeeds.

        Raises:
            httpx.HTTPStatusError: If the HTTP request returns a 4xx or 5xx status code.
            RuntimeError: If all retry attempts are exhausted.
            Exception: For other unexpected errors (only raised on the final attempt).
        """
        client = await cls._get_client()

        # Use provided parameters or fall back to defaults
        use_max_attempts = max_attempts or DEFAULT_MAX_ATTEMPTS
        use_retry_delay = retry_delay or DEFAULT_RETRY_DELAY
        use_timeout = httpx.Timeout(
            connect=timeout or cls._connect_timeout,
            read=None,  # Maintain disabled read timeout for long-running operations
            write=None,
            pool=None,
        )

        # Execute request with retries
        for attempt in range(use_max_attempts):
            attempt_num = attempt + 1
            try:
                response = await client.request(method=method, url=url, json=payload or {}, timeout=use_timeout)
                response.raise_for_status()  # Raise exception for HTTP errors (4xx/5xx)
                logger.debug(f"Request to {url} succeeded (attempt {attempt_num}/{use_max_attempts})")
                return response.json()

            # Handle specific HTTP exceptions
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error for {url} (attempt {attempt_num}/{use_max_attempts}): {e}")
                raise  # Do not retry on HTTP status errors

            except (
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                httpx.ConnectError,
                httpx.TimeoutException,
            ) as e:
                logger.warning(f"Request error for {url} (attempt {attempt_num}/{use_max_attempts}): {e}")

            # Handle unexpected exceptions
            except Exception as e:
                logger.error(f"Unknown error for {url} (attempt {attempt_num}/{use_max_attempts}): {e}")
                if attempt == use_max_attempts - 1:
                    raise  # Raise on the final attempt

            # Exponential backoff for retries
            if attempt < use_max_attempts - 1:
                sleep_time = use_retry_delay * (2**attempt)
                logger.debug(f"Retrying request to {url} in {sleep_time:.2f} seconds (attempt {attempt_num + 1}/{use_max_attempts})")
                await asyncio.sleep(sleep_time)

        # All retry attempts failed
        raise RuntimeError(f"Request to {url} failed after {use_max_attempts} attempts")

    @classmethod
    async def close(cls):
        """
        Closes the thread-local AsyncClient instance.
        This should be called when the thread is done using the client to free resources.
        """
        if hasattr(_thread_local, "client"):
            await _thread_local.client.aclose()
            delattr(_thread_local, "client")
            logger.info("Thread-local async HTTP client closed successfully")
