"""
Kalshi API Client

This module provides a client for interacting with Kalshi's Trade API v2
to fetch market metadata and price history.

API Documentation: https://docs.kalshi.com

Endpoints Used:
    - GET /markets: List all markets
    - GET /markets/{ticker}: Single market details
    - GET /series/{series_ticker}/markets/{ticker}/candlesticks: Price history
    - GET /markets/trades: Trade history

Notes:
    - No authentication required for read-only operations
    - Rate limit: ~3 requests/second recommended
    - Prices are in cents (0-100), need to divide by 100
    - Kalshi organizes markets into "series" (e.g., INXD for S&P daily)
"""

import logging
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional

from data.base_client import BaseMarketClient, RateLimitConfig
from data.schema import (
    MarketMetadata,
    Platform,
    PricePoint,
    normalize_price,
    parse_timestamp,
    map_kalshi_status,
)

logger = logging.getLogger(__name__)


class KalshiClient(BaseMarketClient):
    """
    Client for Kalshi Trade API v2.

    Fetches market data and price history from Kalshi's public API.
    Note: Despite the 'elections' subdomain, this provides access to ALL markets.

    Example:
        async with KalshiClient() as client:
            async for market in client.iterate_all_markets():
                print(market.question)
    """

    # API Base URL
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(
        self,
        rate_limit: Optional[RateLimitConfig] = None,
        **kwargs,
    ):
        """
        Initialize Kalshi client.

        Args:
            rate_limit: Rate limiting configuration
            **kwargs: Additional arguments passed to base class
        """
        if rate_limit is None:
            rate_limit = RateLimitConfig(
                requests_per_second=3.0,  # More conservative for Kalshi
                burst_limit=5,
                retry_attempts=3,
            )

        super().__init__(
            platform=Platform.KALSHI,
            base_url=self.BASE_URL,
            rate_limit=rate_limit,
            **kwargs,
        )

        # Cache for series metadata
        self._series_cache: Dict[str, Dict] = {}

    async def fetch_markets(
        self,
        status: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
        series_ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch markets from Kalshi API.

        Args:
            status: Filter by status ('open', 'closed', 'settled')
            cursor: Pagination cursor
            limit: Maximum markets per request (max 1000)
            series_ticker: Filter by series ticker

        Returns:
            Dictionary with 'markets' list and 'cursor' for next page
        """
        params = {
            "limit": min(limit, 1000),  # Kalshi max is 1000
        }

        if cursor:
            params["cursor"] = cursor

        if status:
            params["status"] = status

        if series_ticker:
            params["series_ticker"] = series_ticker

        try:
            response = await self.get("/markets", params=params)

            return {
                "markets": response.get("markets", []),
                "cursor": response.get("cursor", ""),
            }

        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            raise

    async def fetch_active_series(
        self,
        min_volume: int = 100,
        max_series_to_check: int = 50,
    ) -> List[str]:
        """
        Find series that have markets with trading volume.

        Args:
            min_volume: Minimum total volume across series markets
            max_series_to_check: Maximum number of series to check (for speed)

        Returns:
            List of series tickers with active trading
        """
        try:
            # Get series (limit for speed)
            response = await self.get("/series", params={"limit": max_series_to_check})
            series_list = response.get("series", [])

            active_series = []

            for s in series_list:
                ticker = s.get("ticker", "")

                # Check if series has markets with volume
                markets_resp = await self.fetch_markets(
                    series_ticker=ticker, limit=20
                )
                markets = markets_resp.get("markets", [])

                total_vol = sum(m.get("volume", 0) for m in markets)
                if total_vol >= min_volume:
                    active_series.append(ticker)
                    logger.debug(f"Found active series: {ticker} (vol={total_vol})")

                # Stop if we have enough active series
                if len(active_series) >= 20:
                    break

            logger.info(f"Found {len(active_series)} series with volume >= {min_volume}")
            return active_series

        except Exception as e:
            logger.warning(f"Failed to fetch active series: {e}")
            return []

    async def iterate_markets_with_volume(
        self,
        max_markets: Optional[int] = None,
    ) -> "AsyncIterator[MarketMetadata]":
        """
        Iterate through markets that have trading volume.

        This method finds active series and fetches markets from them,
        filtering for markets with actual trading activity.

        Args:
            max_markets: Maximum total markets to yield

        Yields:
            Normalized MarketMetadata objects for markets with volume
        """
        # Find series with trading activity
        active_series = await self.fetch_active_series(min_volume=100)

        if not active_series:
            logger.warning("No active series found, falling back to default fetch")
            async for market in self.iterate_all_markets(max_markets=max_markets):
                yield market
            return

        total_yielded = 0

        for series_ticker in active_series:
            if max_markets and total_yielded >= max_markets:
                break

            cursor = None
            while True:
                if max_markets and total_yielded >= max_markets:
                    break

                response = await self.fetch_markets(
                    series_ticker=series_ticker,
                    cursor=cursor,
                    limit=100,
                )

                markets = response.get("markets", [])
                if not markets:
                    break

                for raw_market in markets:
                    # Only include markets with some trading volume
                    if raw_market.get("volume", 0) > 0:
                        try:
                            market = self.normalize_market(raw_market)
                            yield market
                            total_yielded += 1

                            if max_markets and total_yielded >= max_markets:
                                return

                        except Exception as e:
                            logger.warning(f"Failed to normalize market: {e}")
                            continue

                cursor = response.get("cursor")
                if not cursor:
                    break

    async def fetch_price_history(
        self,
        market_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        fidelity_minutes: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Fetch candlestick price history from Kalshi API.

        Args:
            market_id: Market ticker
            start_time: Start of time range
            end_time: End of time range
            fidelity_minutes: Period interval (1, 60, or 1440)

        Returns:
            List of candlestick data points

        Notes:
            - Kalshi requires series_ticker for candlestick endpoint
            - Kalshi requires start_ts parameter
            - Valid intervals: 1 (minute), 60 (hour), 1440 (day)
        """
        # Map fidelity to valid Kalshi intervals
        valid_intervals = {1: 1, 60: 60, 1440: 1440}
        interval = min(
            valid_intervals.keys(),
            key=lambda x: abs(x - fidelity_minutes),
        )

        # Get series ticker (required for candlestick endpoint)
        series_ticker = await self._get_series_ticker(market_id)
        if not series_ticker:
            logger.warning(f"Could not find series_ticker for {market_id}")
            return []

        # start_ts is required by Kalshi API
        # Kalshi limits to 5000 candlesticks per request
        # For hourly data: 5000 hours = ~208 days
        # For minute data: 5000 minutes = ~3.5 days
        # For daily data: 5000 days = ~13.7 years
        now = datetime.utcnow()

        if end_time is None:
            end_time = now

        if start_time is None:
            # Default based on interval to stay under 5000 limit
            if interval == 1:
                # Minute data: go back 3 days
                start_time = end_time - timedelta(days=3)
            elif interval == 60:
                # Hourly data: go back 200 days
                start_time = end_time - timedelta(days=200)
            else:
                # Daily data: go back 5 years
                start_time = end_time - timedelta(days=1825)

        params = {
            "period_interval": interval,
            "start_ts": int(start_time.timestamp()),
            "end_ts": int(end_time.timestamp()),
        }

        try:
            endpoint = f"/series/{series_ticker}/markets/{market_id}/candlesticks"
            response = await self.get(endpoint, params=params)

            candlesticks = response.get("candlesticks", [])
            logger.debug(f"Fetched {len(candlesticks)} candlesticks for {market_id}")
            return candlesticks

        except Exception as e:
            logger.warning(f"Failed to fetch candlesticks for {market_id}: {e}")
            return []

    async def fetch_trades(
        self,
        ticker: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch trade history.

        Args:
            ticker: Filter by market ticker
            min_ts: Minimum timestamp (Unix)
            max_ts: Maximum timestamp (Unix)
            limit: Maximum trades per request
            cursor: Pagination cursor

        Returns:
            Dictionary with 'trades' list and 'cursor'
        """
        params = {"limit": limit}

        if ticker:
            params["ticker"] = ticker
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor

        try:
            response = await self.get("/markets/trades", params=params)
            return {
                "trades": response.get("trades", []),
                "cursor": response.get("cursor", ""),
            }
        except Exception as e:
            logger.warning(f"Failed to fetch trades: {e}")
            return {"trades": [], "cursor": ""}

    async def fetch_series(self, series_ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch series metadata.

        Args:
            series_ticker: Series identifier

        Returns:
            Series metadata or None
        """
        if series_ticker in self._series_cache:
            return self._series_cache[series_ticker]

        try:
            response = await self.get(f"/series/{series_ticker}")
            series = response.get("series", response)
            self._series_cache[series_ticker] = series
            return series
        except Exception as e:
            logger.debug(f"Failed to fetch series {series_ticker}: {e}")
            return None

    async def _get_series_ticker(self, market_ticker: str) -> Optional[str]:
        """
        Get series ticker for a market.

        Args:
            market_ticker: Market ticker

        Returns:
            Series ticker or None
        """
        # Check cache first (populated during market fetch)
        cache_key = f"market_{market_ticker}_series"
        if cache_key in self._series_cache:
            return self._series_cache[cache_key]

        # Try to get from market details
        try:
            response = await self.get(f"/markets/{market_ticker}")
            market = response.get("market", {})
            series_ticker = market.get("series_ticker")
            if series_ticker:
                self._series_cache[cache_key] = series_ticker
                return series_ticker
        except Exception as e:
            logger.debug(f"Failed to get market details for {market_ticker}: {e}")

        # Try common patterns (e.g., INXD-24JAN01 -> INXD)
        if "-" in market_ticker:
            potential_series = market_ticker.split("-")[0]
            series = await self.fetch_series(potential_series)
            if series:
                self._series_cache[cache_key] = potential_series
                return potential_series

        return None

    def cache_series_ticker(self, market_ticker: str, series_ticker: str) -> None:
        """Cache series ticker for a market."""
        if series_ticker:
            cache_key = f"market_{market_ticker}_series"
            self._series_cache[cache_key] = series_ticker

    def normalize_market(self, raw: Dict[str, Any]) -> MarketMetadata:
        """
        Convert Kalshi API market response to unified schema.

        Args:
            raw: Raw market data from API

        Returns:
            Normalized MarketMetadata object
        """
        ticker = raw.get("ticker", "")
        market_id = f"kalshi_{ticker}"

        # Determine if binary
        market_type = raw.get("market_type", "").lower()
        is_binary = market_type == "binary"

        # Extract series_ticker and cache it for price fetching
        series_ticker = raw.get("series_ticker", "")
        if series_ticker:
            self.cache_series_ticker(ticker, series_ticker)

        # Extract category from series or event
        category = raw.get("category", "")
        if not category and series_ticker:
            category = series_ticker

        # Build question from title and subtitle
        title = raw.get("title", "")
        subtitle = raw.get("yes_sub_title", raw.get("subtitle", ""))
        question = title
        if subtitle and subtitle != title:
            question = f"{title}: {subtitle}"

        return MarketMetadata(
            market_id=market_id,
            platform=Platform.KALSHI,
            slug=ticker,
            question=question,
            description=raw.get("rules_primary"),
            category=category or "uncategorized",
            series_ticker=raw.get("series_ticker"),
            tags=raw.get("tags", []),
            created_at=parse_timestamp(raw.get("created_time")),
            open_time=parse_timestamp(raw.get("open_time")),
            close_time=parse_timestamp(raw.get("close_time")),
            resolution_time=parse_timestamp(raw.get("settlement_time")),
            is_binary=is_binary,
            outcome_yes_id=ticker,  # Kalshi uses ticker for trading
            outcome_no_id=None,
            status=map_kalshi_status(raw.get("status", "open")),
            result=self._extract_result(raw),
            min_tick_size=self._cents_to_prob(raw.get("tick_size", 1)),
            volume_total=self._parse_float(raw.get("volume")),
            liquidity=self._cents_to_usd(raw.get("liquidity")),
            source_hash=MarketMetadata.compute_hash(raw),
        )

    def normalize_price_point(
        self,
        raw: Dict[str, Any],
        market_id: str,
    ) -> PricePoint:
        """
        Convert Kalshi candlestick response to unified schema.

        Args:
            raw: Raw candlestick data from API
            market_id: Parent market ID

        Returns:
            Normalized PricePoint object

        Notes:
            - Kalshi prices are in cents (0-100)
            - Candlestick structure includes yes_price, yes_bid, yes_ask
        """
        # Parse timestamp
        ts = raw.get("end_period_ts") or raw.get("ts")
        timestamp = datetime.utcfromtimestamp(ts) if ts else datetime.utcnow()

        # Extract price data (all in cents)
        price_data = raw.get("price", raw.get("yes_price", {}))
        if isinstance(price_data, dict):
            close_price = price_data.get("close")
            open_price = price_data.get("open")
            high_price = price_data.get("high")
            low_price = price_data.get("low")
        else:
            close_price = price_data
            open_price = high_price = low_price = None

        # Extract bid/ask
        yes_bid = raw.get("yes_bid", {})
        yes_ask = raw.get("yes_ask", {})

        bid_close = yes_bid.get("close") if isinstance(yes_bid, dict) else yes_bid
        ask_close = yes_ask.get("close") if isinstance(yes_ask, dict) else yes_ask

        # Convert cents to probabilities
        # Note: Don't default to 0.5 - let caller filter out None prices
        price_yes = self._cents_to_prob(close_price)
        bid_yes = self._cents_to_prob(bid_close)
        ask_yes = self._cents_to_prob(ask_close)

        # If no close price but we have bid/ask, use midpoint
        if price_yes is None and bid_yes is not None and ask_yes is not None:
            price_yes = (bid_yes + ask_yes) / 2

        # Compute spread and midpoint
        spread = None
        midpoint = price_yes if price_yes is not None else 0.5
        if bid_yes is not None and ask_yes is not None:
            spread = ask_yes - bid_yes
            midpoint = (bid_yes + ask_yes) / 2

        # Get volume - if 0 and no price data, this is a placeholder candle
        volume = self._parse_float(raw.get("volume"))
        has_trade_data = (
            price_yes is not None or
            (volume is not None and volume > 0) or
            open_price is not None
        )

        # If no actual trade data, return None to signal caller to skip
        if not has_trade_data:
            return None

        # Use midpoint as fallback for price_yes if we have bid/ask
        if price_yes is None:
            price_yes = midpoint

        return PricePoint(
            timestamp=timestamp,
            market_id=market_id,
            platform=Platform.KALSHI,
            price_yes=price_yes,
            price_no=1.0 - price_yes,
            bid_yes=bid_yes,
            ask_yes=ask_yes,
            bid_no=1.0 - ask_yes if ask_yes is not None else None,
            ask_no=1.0 - bid_yes if bid_yes is not None else None,
            midpoint=midpoint,
            spread=spread,
            volume=volume,
            volume_usd=None,
            open_interest=self._parse_float(raw.get("open_interest")),
            liquidity_usd=None,
            open=self._cents_to_prob(open_price),
            high=self._cents_to_prob(high_price),
            low=self._cents_to_prob(low_price),
            close=self._cents_to_prob(close_price),
            fidelity_minutes=60,
            source="api",
        )

    def _extract_result(self, raw: Dict[str, Any]) -> Optional[str]:
        """Extract resolution result from market data."""
        result = raw.get("result")
        if result:
            return result.lower()

        # Check settlement value
        settlement = raw.get("settlement_value")
        if settlement is not None:
            return "yes" if settlement > 50 else "no"

        return None

    @staticmethod
    def _cents_to_prob(cents: Any) -> Optional[float]:
        """Convert cents (0-100) to probability (0-1)."""
        if cents is None:
            return None
        try:
            value = float(cents)
            if value > 1:
                return value / 100.0
            return value
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _cents_to_usd(cents: Any) -> Optional[float]:
        """Convert cents to USD."""
        if cents is None:
            return None
        try:
            return float(cents) / 100.0
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_float(value: Any) -> Optional[float]:
        """Safely parse float value."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    async def get_market_details(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information for a specific market.

        Args:
            ticker: Market ticker

        Returns:
            Market details or None
        """
        try:
            response = await self.get(f"/markets/{ticker}")
            return response.get("market", response)
        except Exception as e:
            logger.warning(f"Failed to fetch market details for {ticker}: {e}")
            return None

    async def get_all_series(self) -> List[Dict[str, Any]]:
        """
        Fetch all series (market categories).

        Returns:
            List of series objects
        """
        try:
            response = await self.get("/series")
            return response.get("series", [])
        except Exception as e:
            logger.warning(f"Failed to fetch series: {e}")
            return []

    async def get_events(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Fetch events (groups of related markets).

        Args:
            status: Filter by status
            limit: Maximum events

        Returns:
            List of event objects
        """
        params = {"limit": limit}
        if status:
            params["status"] = status

        try:
            response = await self.get("/events", params=params)
            return response.get("events", [])
        except Exception as e:
            logger.warning(f"Failed to fetch events: {e}")
            return []
