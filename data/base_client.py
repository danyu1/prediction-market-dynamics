"""
Base API Client Infrastructure

This module provides an abstract base class for prediction market API clients
with built-in rate limiting, retry logic, and async HTTP support.

Features:
    - Semaphore-based concurrency control
    - Minimum interval between requests
    - Exponential backoff retry logic
    - Async HTTP with aiohttp
    - Configurable rate limits per platform

Usage:
    Subclass BaseMarketClient and implement:
        - fetch_markets()
        - fetch_price_history()
        - normalize_market()
        - normalize_price_point()
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

from data.schema import MarketMetadata, PricePoint, Platform

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """
    Rate limiting configuration for API clients.

    Attributes:
        requests_per_second: Maximum requests per second
        burst_limit: Maximum concurrent requests (semaphore size)
        retry_attempts: Number of retry attempts on failure
        retry_backoff_base: Base for exponential backoff (seconds)
        retry_backoff_max: Maximum backoff time (seconds)
    """
    requests_per_second: float = 5.0
    burst_limit: int = 10
    retry_attempts: int = 3
    retry_backoff_base: float = 2.0
    retry_backoff_max: float = 60.0


@dataclass
class RequestStats:
    """Track request statistics for monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    rate_limited_requests: int = 0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests


class RateLimiter:
    """
    Token bucket rate limiter with async support.

    Ensures requests don't exceed the configured rate limit
    while allowing for burst traffic up to the burst limit.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._semaphore = asyncio.Semaphore(config.burst_limit)
        self._min_interval = 1.0 / config.requests_per_second
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        async with self._semaphore:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_request_time
                wait_time = self._min_interval - elapsed

                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                self._last_request_time = time.monotonic()


class BaseMarketClient(ABC):
    """
    Abstract base class for prediction market API clients.

    Provides common infrastructure for rate limiting, retries,
    and async HTTP requests. Subclasses must implement platform-specific
    methods for fetching and normalizing data.

    Attributes:
        platform: The platform this client connects to
        base_url: Base URL for API requests
        rate_limit: Rate limiting configuration
        stats: Request statistics for monitoring
    """

    def __init__(
        self,
        platform: Platform,
        base_url: str,
        rate_limit: Optional[RateLimitConfig] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        """
        Initialize the base client.

        Args:
            platform: Target platform
            base_url: Base URL for API requests
            rate_limit: Rate limiting configuration (uses defaults if None)
            session: Optional aiohttp session (created if not provided)
        """
        self.platform = platform
        self.base_url = base_url.rstrip("/")
        self.rate_limit = rate_limit or RateLimitConfig()
        self._session = session
        self._owns_session = session is None
        self._rate_limiter = RateLimiter(self.rate_limit)
        self.stats = RequestStats()

    async def __aenter__(self) -> "BaseMarketClient":
        """Async context manager entry."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._owns_session = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure a session exists, creating one if necessary."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._owns_session = True
        return self._session

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a rate-limited HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (appended to base_url)
            params: Query parameters
            headers: Additional headers
            **kwargs: Additional arguments passed to aiohttp

        Returns:
            JSON response as dictionary

        Raises:
            aiohttp.ClientError: On persistent request failure
        """
        url = f"{self.base_url}{endpoint}"
        session = await self._ensure_session()

        for attempt in range(self.rate_limit.retry_attempts):
            await self._rate_limiter.acquire()
            self.stats.total_requests += 1

            start_time = time.monotonic()

            try:
                async with session.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    **kwargs,
                ) as response:
                    elapsed_ms = (time.monotonic() - start_time) * 1000

                    # Handle rate limiting
                    if response.status == 429:
                        self.stats.rate_limited_requests += 1
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(
                            f"Rate limited on {url}, waiting {retry_after}s"
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    # Handle server errors with retry
                    if response.status >= 500:
                        self.stats.retried_requests += 1
                        backoff = min(
                            self.rate_limit.retry_backoff_base ** attempt,
                            self.rate_limit.retry_backoff_max,
                        )
                        logger.warning(
                            f"Server error {response.status} on {url}, "
                            f"retrying in {backoff:.1f}s (attempt {attempt + 1})"
                        )
                        await asyncio.sleep(backoff)
                        continue

                    # Raise for client errors
                    response.raise_for_status()

                    # Success
                    self.stats.successful_requests += 1
                    self.stats.total_latency_ms += elapsed_ms

                    return await response.json()

            except aiohttp.ClientError as e:
                self.stats.retried_requests += 1
                if attempt == self.rate_limit.retry_attempts - 1:
                    self.stats.failed_requests += 1
                    logger.error(f"Request failed after {attempt + 1} attempts: {e}")
                    raise

                backoff = min(
                    self.rate_limit.retry_backoff_base ** attempt,
                    self.rate_limit.retry_backoff_max,
                )
                logger.warning(
                    f"Request error: {e}, retrying in {backoff:.1f}s "
                    f"(attempt {attempt + 1})"
                )
                await asyncio.sleep(backoff)

        # Should not reach here, but handle edge case
        self.stats.failed_requests += 1
        raise aiohttp.ClientError(f"Request to {url} failed after all retries")

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convenience method for GET requests."""
        return await self._request("GET", endpoint, params=params, **kwargs)

    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convenience method for POST requests."""
        return await self._request("POST", endpoint, json=data, **kwargs)

    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    async def fetch_markets(
        self,
        status: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Fetch paginated list of markets.

        Args:
            status: Filter by market status
            cursor: Pagination cursor
            limit: Maximum markets per page

        Returns:
            Dictionary with 'markets' list and optional 'cursor' for next page
        """
        pass

    @abstractmethod
    async def fetch_price_history(
        self,
        market_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        fidelity_minutes: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Fetch price history for a specific market.

        Args:
            market_id: Market identifier (token ID or ticker)
            start_time: Start of time range
            end_time: End of time range
            fidelity_minutes: Data resolution (1, 60, 1440)

        Returns:
            List of raw price data points
        """
        pass

    @abstractmethod
    def normalize_market(self, raw: Dict[str, Any]) -> MarketMetadata:
        """
        Convert platform-specific market data to unified schema.

        Args:
            raw: Raw API response for a single market

        Returns:
            Normalized MarketMetadata object
        """
        pass

    @abstractmethod
    def normalize_price_point(
        self,
        raw: Dict[str, Any],
        market_id: str,
    ) -> PricePoint:
        """
        Convert platform-specific price data to unified schema.

        Args:
            raw: Raw API response for a single price point
            market_id: Parent market identifier

        Returns:
            Normalized PricePoint object
        """
        pass

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def iterate_all_markets(
        self,
        status: Optional[str] = None,
        limit_per_page: int = 100,
        max_markets: Optional[int] = None,
    ) -> AsyncIterator[MarketMetadata]:
        """
        Iterate through all markets with automatic pagination.

        Args:
            status: Filter by market status
            limit_per_page: Markets per API request
            max_markets: Maximum total markets to fetch

        Yields:
            Normalized MarketMetadata objects
        """
        cursor = None
        total_fetched = 0

        while True:
            response = await self.fetch_markets(
                status=status,
                cursor=cursor,
                limit=limit_per_page,
            )

            markets = response.get("markets", response.get("data", []))
            if not markets:
                break

            for raw_market in markets:
                try:
                    market = self.normalize_market(raw_market)
                    yield market
                    total_fetched += 1

                    if max_markets and total_fetched >= max_markets:
                        return

                except Exception as e:
                    logger.warning(f"Failed to normalize market: {e}")
                    continue

            cursor = response.get("cursor", response.get("next_cursor"))
            if not cursor:
                break

    async def fetch_and_normalize_prices(
        self,
        market: MarketMetadata,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        fidelity_minutes: int = 60,
    ) -> List[PricePoint]:
        """
        Fetch and normalize price history for a market.

        Args:
            market: Market metadata object
            start_time: Start of time range
            end_time: End of time range
            fidelity_minutes: Data resolution

        Returns:
            List of normalized PricePoint objects
        """
        token_id = market.outcome_yes_id
        if not token_id:
            logger.warning(f"No token ID for market {market.market_id}")
            return []

        try:
            raw_prices = await self.fetch_price_history(
                market_id=token_id,
                start_time=start_time,
                end_time=end_time,
                fidelity_minutes=fidelity_minutes,
            )

            prices = []
            for raw in raw_prices:
                try:
                    price = self.normalize_price_point(raw, market.market_id)
                    # Skip None values (returned when no actual trade data)
                    if price is not None:
                        prices.append(price)
                except Exception as e:
                    logger.debug(f"Failed to normalize price point: {e}")
                    continue

            return prices

        except Exception as e:
            logger.error(f"Failed to fetch prices for {market.market_id}: {e}")
            return []

    def get_stats_summary(self) -> str:
        """Get a formatted summary of request statistics."""
        return (
            f"Requests: {self.stats.total_requests} total, "
            f"{self.stats.successful_requests} success, "
            f"{self.stats.failed_requests} failed, "
            f"{self.stats.retried_requests} retried, "
            f"{self.stats.rate_limited_requests} rate-limited | "
            f"Success rate: {self.stats.success_rate:.1f}% | "
            f"Avg latency: {self.stats.avg_latency_ms:.0f}ms"
        )
