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
import os
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

_DEFAULT_MAX_CONNECTIONS = 256
_DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 64
_DEFAULT_KEEPALIVE_EXPIRY = 30.0


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning(f"Invalid {name}={raw!r}, fallback to default={default}")
        return default
    if parsed <= 0:
        logger.warning(f"Invalid {name}={raw!r}, expected positive integer, fallback to default={default}")
        return default
    return parsed


def _parse_non_negative_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        parsed = float(raw)
    except ValueError:
        logger.warning(f"Invalid {name}={raw!r}, fallback to default={default}")
        return default
    if parsed < 0:
        logger.warning(f"Invalid {name}={raw!r}, expected non-negative float, fallback to default={default}")
        return default
    return parsed


def _load_http_client_limits() -> tuple[int, int, float]:
    max_conn = _parse_positive_int_env("SIIRL_HTTP_MAX_CONNECTIONS", _DEFAULT_MAX_CONNECTIONS)
    max_keepalive = _parse_positive_int_env("SIIRL_HTTP_MAX_KEEPALIVE_CONNECTIONS", _DEFAULT_MAX_KEEPALIVE_CONNECTIONS)
    keepalive_expiry = _parse_non_negative_float_env("SIIRL_HTTP_KEEPALIVE_EXPIRY", _DEFAULT_KEEPALIVE_EXPIRY)
    if max_keepalive > max_conn:
        logger.warning(
            f"SIIRL_HTTP_MAX_KEEPALIVE_CONNECTIONS={max_keepalive} exceeds "
            f"SIIRL_HTTP_MAX_CONNECTIONS={max_conn}, clamp keepalive to {max_conn}"
        )
        max_keepalive = max_conn
    return max_conn, max_keepalive, keepalive_expiry


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
    _metrics_lock = threading.Lock()
    _metrics = {
        "attempts": 0,
        "success": 0,
        "timeouts": 0,
        "errors": 0,
    }

    @classmethod
    def _record_metrics(cls, *, attempts: int = 0, success: int = 0, timeouts: int = 0, errors: int = 0) -> None:
        if attempts == 0 and success == 0 and timeouts == 0 and errors == 0:
            return
        with cls._metrics_lock:
            cls._metrics["attempts"] += int(attempts)
            cls._metrics["success"] += int(success)
            cls._metrics["timeouts"] += int(timeouts)
            cls._metrics["errors"] += int(errors)

    @classmethod
    def drain_metrics(cls) -> dict[str, int]:
        with cls._metrics_lock:
            current = dict(cls._metrics)
            cls._metrics = {
                "attempts": 0,
                "success": 0,
                "timeouts": 0,
                "errors": 0,
            }
        return current

    @classmethod
    async def _get_client(cls) -> httpx.AsyncClient:
        """
        Retrieves or initializes the thread-local AsyncClient instance.

        Returns:
            httpx.AsyncClient: The thread-local HTTP client instance.
        """
        if not hasattr(_thread_local, "client"):
            max_conn, max_keepalive, keepalive_expiry = _load_http_client_limits()
            _thread_local.client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=max_conn,
                    max_keepalive_connections=max_keepalive,
                    keepalive_expiry=keepalive_expiry,
                ),
                timeout=httpx.Timeout(
                    connect=cls._connect_timeout,
                    read=None,  # Disable read timeout for long-running operations (e.g., model generation)
                    write=None,
                    pool=None,
                ),
                http2=False,
                follow_redirects=True,
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
            httpx.HTTPStatusError: For non-transient HTTP status errors.
            RuntimeError: If retries are exhausted.
            asyncio.CancelledError: If the request task is cancelled.
        """
        client = await cls._get_client()
        attempts = 0
        success = 0
        timeouts = 0
        errors = 0

        # Use provided parameters or fall back to defaults
        use_max_attempts = max_attempts or DEFAULT_MAX_ATTEMPTS
        use_retry_delay = retry_delay or DEFAULT_RETRY_DELAY
        use_timeout = httpx.Timeout(
            connect=timeout or cls._connect_timeout,
            read=None,  # No read timeout for long generation.
            write=None,
            pool=None,
        )

        try:
            for attempt in range(use_max_attempts):
                attempt_num = attempt + 1
                attempts += 1
                try:
                    response = await client.request(method=method, url=url, json=payload or {}, timeout=use_timeout)
                    response.raise_for_status()
                    success += 1
                    logger.debug(f"Request to {url} succeeded (attempt {attempt_num}/{use_max_attempts})")
                    return response.json()

                except httpx.HTTPStatusError as e:
                    errors += 1
                    status_code = e.response.status_code if e.response is not None else None
                    if status_code in {408, 425, 429, 500, 502, 503, 504}:
                        logger.warning(
                            f"Transient HTTP error for {url} (attempt {attempt_num}/{use_max_attempts}): "
                            f"status={status_code}, detail={e!r}"
                        )
                    else:
                        logger.error(
                            f"HTTP error for {url} (attempt {attempt_num}/{use_max_attempts}): " f"status={status_code}, detail={e!r}"
                        )
                        raise

                except httpx.ConnectTimeout as e:
                    timeouts += 1
                    errors += 1
                    logger.warning(
                        f"Request error for {url} (attempt {attempt_num}/{use_max_attempts}): " f"{type(e).__name__}, detail={e!r}"
                    )

                except httpx.ReadTimeout as e:
                    timeouts += 1
                    errors += 1
                    logger.warning(
                        f"Request error for {url} (attempt {attempt_num}/{use_max_attempts}): " f"{type(e).__name__}, detail={e!r}"
                    )

                except httpx.TimeoutException as e:
                    timeouts += 1
                    errors += 1
                    logger.warning(
                        f"Request error for {url} (attempt {attempt_num}/{use_max_attempts}): " f"{type(e).__name__}, detail={e!r}"
                    )

                except (httpx.ConnectError, httpx.RequestError) as e:
                    errors += 1
                    logger.warning(
                        f"Request error for {url} (attempt {attempt_num}/{use_max_attempts}): " f"{type(e).__name__}, detail={e!r}"
                    )

                except asyncio.CancelledError:
                    raise

                except Exception as e:
                    errors += 1
                    logger.error(
                        f"Unknown error for {url} (attempt {attempt_num}/{use_max_attempts}): " f"{type(e).__name__}, detail={e!r}"
                    )
                    if attempt == use_max_attempts - 1:
                        raise

                if attempt < use_max_attempts - 1:
                    sleep_time = use_retry_delay * (2**attempt)
                    logger.debug(
                        f"Retrying request to {url} in {sleep_time:.2f} seconds " f"(attempt {attempt_num + 1}/{use_max_attempts})"
                    )
                    await asyncio.sleep(sleep_time)

            raise RuntimeError(f"Request to {url} failed after {use_max_attempts} attempts")
        finally:
            cls._record_metrics(attempts=attempts, success=success, timeouts=timeouts, errors=errors)

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
