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
import time
import ray 
import asyncio
import time
import multiprocessing
import requests
from loguru import logger
from typing import Optional, Dict, Any, Literal
from requests.exceptions import RequestException
import httpx
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
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
    """Global asynchronous HTTP client with singleton pattern and auto-initialization.
    No manual parameter passing required for usage.
    """
    _instance: Optional[httpx.AsyncClient] = None
    _lock = asyncio.Lock()
    _max_connections: int = 50
    _connect_timeout: float = 10.0

    # -------------------------- Singleton Initialization --------------------------
    @classmethod
    async def _get_client(cls) -> httpx.AsyncClient:
        """Internal method to get client instance (auto-initialize and reuse).
        
        Returns:
            Initialized httpx.AsyncClient instance
        """
        async with cls._lock:
            if cls._instance is None:
                # Create async HTTP client
                cls._instance = httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=None,  # Fix: Change None to reasonable value in production
                    ),
                    timeout=httpx.Timeout(
                        connect=cls._connect_timeout,
                        read=None,  # Disable read timeout for large model generation
                        write=None,
                        pool=None
                    ),
                    http2=False,
                    follow_redirects=True,
                )
            return cls._instance

    # -------------------------- Core Request Method --------------------------
    @classmethod
    async def make_request(
        cls,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        method: HTTPMethod = "POST",
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make asynchronous HTTP request with exponential backoff retry mechanism.
        
        Args:
            url: Target URL for the request
            payload: Optional JSON payload for the request
            method: HTTP method (GET/POST/PUT/DELETE)
            timeout: Request timeout in seconds (only affects connect timeout)
            max_attempts: Maximum retry attempts (overrides default)
            retry_delay: Initial retry delay in seconds (overrides default)
        
        Returns:
            JSON response parsed as dictionary if request succeeds
        
        Raises:
            httpx.HTTPStatusError: For HTTP 4xx/5xx errors
            RuntimeError: When all retry attempts are exhausted
            Exception: For other unexpected errors (only on final attempt)
        """
        client = await cls._get_client()

        # Parameter fallback (compatible with instance attributes/default values)
        use_max_attempts = max_attempts or getattr(cls, "max_attempts", DEFAULT_MAX_ATTEMPTS)
        use_retry_delay = retry_delay or getattr(cls, "retry_delay", DEFAULT_RETRY_DELAY)
        # Note: Client has disabled read timeout, timeout here only overrides connect timeout
        use_timeout = httpx.Timeout(
            connect=timeout or cls._connect_timeout,
            read=None,  # Force disable read timeout for large model generation
            write=None,
            pool=None
        )

        # Core request + retry logic
        for attempt in range(use_max_attempts):
            attempt_num = attempt + 1
            try:
                # Send HTTP request
                response = await client.request(
                    method=method,
                    url=url,
                    json=payload or {},
                    timeout=use_timeout
                )
                # Raise exception for HTTP 4xx/5xx status codes
                response.raise_for_status()
                logger.debug(f"Request to {url} succeeded (attempt {attempt_num}/{use_max_attempts}), status code: {response.status_code}")
                return response.json()

            # -------------------------- Hierarchical Exception Handling (httpx async client compatible) --------------------------
            except httpx.HTTPStatusError as e:
                # HTTP errors (4xx/5xx): no retry, detailed logging
                error_note = f"Status code: {e.response.status_code}, Response text: {e.response.text if e.response else 'None'}"
                logger.error(
                    f"HTTP error for request to {url} (attempt {attempt_num}/{use_max_attempts}): {str(e)}\n{error_note}"
                )
                raise  # Raise exception without retry

            except httpx.ReadTimeout as e:
                # Read timeout (most common for large model generation)
                logger.warning(
                    f"Read timeout for request to {url} (attempt {attempt_num}/{use_max_attempts}): "
                    f"Connect timeout {use_timeout.connect}s, read timeout disabled, error details: {str(e)}"
                )

            except httpx.ConnectTimeout as e:
                # Connection timeout (network/server unreachable)
                logger.warning(
                    f"Connection timeout for request to {url} (attempt {attempt_num}/{use_max_attempts}): "
                    f"Timeout {use_timeout.connect}s, target: {url}, error details: {str(e)}"
                )

            except httpx.ConnectError as e:
                # Connection failure (port unreachable/network interruption)
                logger.warning(
                    f"Connection failed for request to {url} (attempt {attempt_num}/{use_max_attempts}): "
                    f"Target: {url}, error details: {str(e)}"
                )

            except httpx.TimeoutException as e:
                # Other timeout exceptions (fallback)
                logger.warning(
                    f"Timeout for request to {url} (attempt {attempt_num}/{use_max_attempts}): "
                    f"Timeout type: {type(e).__name__}, details: {str(e)}"
                )

            except Exception as e:
                # Unknown exceptions: raise on final attempt, retry otherwise
                logger.error(
                    f"Unknown error for request to {url} (attempt {attempt_num}/{use_max_attempts}): "
                    f"Exception type: {type(e).__name__}, details: {str(e)}"
                )
                if attempt == use_max_attempts - 1:
                    raise  # Raise exception on final attempt

            # -------------------------- Exponential Backoff Retry (async sleep) --------------------------
            if attempt < use_max_attempts - 1:
                sleep_time = use_retry_delay * (2 ** attempt)
                logger.debug(
                    f"Retrying request to {url}, waiting {sleep_time:.2f} seconds before attempt {attempt_num + 1}/{use_max_attempts}"
                )
                await asyncio.sleep(sleep_time)  # Async sleep to avoid blocking event loop

        # All retry attempts exhausted
        raise RuntimeError(
            f"Request to {url} failed: {use_max_attempts} retry attempts exhausted, method: {method}, connect timeout: {use_timeout.connect}s"
        )

    # -------------------------- Resource Release --------------------------
    @classmethod
    async def close(cls):
        """Close the HTTP client (call on program exit)."""
        async with cls._lock:
            if cls._instance is not None:
                await cls._instance.aclose()
                cls._instance = None
                logger.info("Global async HTTP client closed successfully")