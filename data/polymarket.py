"""
Polymarket API Client

This module provides a client for interacting with Polymarket's APIs
to fetch market metadata and price history.

Polymarket has two main APIs:
    - CLOB API (https://clob.polymarket.com): Trading and price data
    - Gamma API (https://gamma-api.polymarket.com): Rich market metadata

Endpoints Used:
    - Gamma /markets: Market discovery with full metadata
    - CLOB /prices-history: Historical price candlesticks
    - CLOB /book: Orderbook snapshots (for spread analysis)

Notes:
    - No authentication required for read-only operations
    - Rate limit: ~10 requests/second recommended
    - Binary markets have exactly 2 tokens (Yes/No)
    - Prices are probabilities in [0, 1]
"""

import logging
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from data.base_client import BaseMarketClient, RateLimitConfig
from data.schema import (
    MarketMetadata,
    MarketStatus,
    Platform,
    PricePoint,
    normalize_price,
    parse_timestamp,
    map_polymarket_status,
)

logger = logging.getLogger(__name__)


class PolymarketClient(BaseMarketClient):
    """
    Client for Polymarket CLOB and Gamma APIs.

    Uses the Gamma API for market discovery (richer metadata) and
    the CLOB API for price history and orderbook data.

    Example:
        async with PolymarketClient() as client:
            async for market in client.iterate_all_markets():
                print(market.question)
    """

    # API Base URLs
    CLOB_BASE_URL = "https://clob.polymarket.com"
    GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(
        self,
        rate_limit: Optional[RateLimitConfig] = None,
        **kwargs,
    ):
        """
        Initialize Polymarket client.

        Args:
            rate_limit: Rate limiting configuration
            **kwargs: Additional arguments passed to base class
        """
        if rate_limit is None:
            rate_limit = RateLimitConfig(
                requests_per_second=5.0,
                burst_limit=10,
                retry_attempts=3,
            )

        super().__init__(
            platform=Platform.POLYMARKET,
            base_url=self.CLOB_BASE_URL,
            rate_limit=rate_limit,
            **kwargs,
        )

        self.gamma_url = self.GAMMA_BASE_URL

    async def fetch_markets(
        self,
        status: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Fetch markets from Gamma API.

        The Gamma API provides richer metadata than the CLOB /markets endpoint,
        including categories, tags, and detailed outcome information.

        Args:
            status: Filter by status ('open', 'closed')
            cursor: Offset for pagination
            limit: Maximum markets per request

        Returns:
            Dictionary with 'markets' list and 'cursor' for next page
        """
        params = {
            "limit": limit,
            "order": "id",
            "ascending": "false",
        }

        if cursor:
            params["offset"] = cursor

        if status:
            if status == "open":
                params["closed"] = "false"
                params["active"] = "true"
            elif status == "closed":
                params["closed"] = "true"

        # Use Gamma API for market discovery
        url = f"{self.gamma_url}/markets"
        session = await self._ensure_session()

        await self._rate_limiter.acquire()
        self.stats.total_requests += 1

        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                self.stats.successful_requests += 1

                # Gamma returns array directly
                markets = data if isinstance(data, list) else data.get("data", [])

                # Calculate next cursor
                next_cursor = None
                if len(markets) == limit:
                    current_offset = int(cursor or 0)
                    next_cursor = str(current_offset + limit)

                return {
                    "markets": markets,
                    "cursor": next_cursor,
                }

        except Exception as e:
            self.stats.failed_requests += 1
            logger.error(f"Failed to fetch markets: {e}")
            raise

    async def fetch_markets_by_volume(
        self,
        cursor: Optional[str] = None,
        limit: int = 100,
        min_volume: float = 1000.0,
    ) -> Dict[str, Any]:
        """
        Fetch markets ordered by volume (most active first).

        Args:
            cursor: Offset for pagination
            limit: Maximum markets per request
            min_volume: Minimum volume to include

        Returns:
            Dictionary with 'markets' list and 'cursor' for next page
        """
        params = {
            "limit": limit,
            "order": "volume",
            "ascending": "false",
            "closed": "false",
            "active": "true",
        }

        if cursor:
            params["offset"] = cursor

        url = f"{self.gamma_url}/markets"
        session = await self._ensure_session()

        await self._rate_limiter.acquire()
        self.stats.total_requests += 1

        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                self.stats.successful_requests += 1

                markets = data if isinstance(data, list) else data.get("data", [])

                # Filter by minimum volume
                markets = [
                    m for m in markets
                    if (m.get("volumeNum") or 0) >= min_volume
                ]

                # Calculate next cursor
                next_cursor = None
                if len(markets) == limit:
                    current_offset = int(cursor or 0)
                    next_cursor = str(current_offset + limit)

                return {
                    "markets": markets,
                    "cursor": next_cursor,
                }

        except Exception as e:
            self.stats.failed_requests += 1
            logger.error(f"Failed to fetch markets by volume: {e}")
            raise

    async def iterate_markets_with_volume(
        self,
        max_markets: Optional[int] = None,
        min_volume: float = 1000.0,
    ) -> AsyncIterator["MarketMetadata"]:
        """
        Iterate through markets that have trading volume.

        Fetches markets ordered by volume (most active first),
        filtering out short-term crypto markets and those with low volume.

        Args:
            max_markets: Maximum total markets to yield
            min_volume: Minimum volume in USD to include

        Yields:
            Normalized MarketMetadata objects for markets with volume
        """
        cursor = None
        total_yielded = 0

        while True:
            if max_markets and total_yielded >= max_markets:
                break

            response = await self.fetch_markets_by_volume(
                cursor=cursor,
                limit=100,
                min_volume=min_volume,
            )

            markets = response.get("markets", [])
            if not markets:
                break

            for raw_market in markets:
                question = raw_market.get("question", "")

                # Skip short-term crypto price prediction markets
                # These have very little historical data
                if self._is_short_term_crypto_market(question):
                    continue

                try:
                    market = self.normalize_market(raw_market)

                    # Only include binary markets with token IDs
                    if market.is_binary and market.outcome_yes_id:
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

    @staticmethod
    def _is_short_term_crypto_market(question: str) -> bool:
        """
        Check if a market is a short-term crypto price prediction.

        These markets (e.g., "Bitcoin Up or Down - January 4, 8:00PM-8:15PM ET")
        have very short durations and minimal price history.

        Args:
            question: Market question text

        Returns:
            True if this is a short-term crypto market to skip
        """
        # Common patterns for short-term crypto markets
        short_term_patterns = [
            "Up or Down",  # Crypto up/down predictions
            "AM ET", "PM ET",  # Time-specific markets
            "AM PT", "PM PT",
        ]

        question_lower = question.lower()
        for pattern in short_term_patterns:
            if pattern.lower() in question_lower:
                return True

        return False

    async def fetch_price_history(
        self,
        market_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        fidelity_minutes: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Fetch price history from CLOB API.

        Args:
            market_id: Token ID (clobTokenId) for the outcome
            start_time: Start of time range
            end_time: End of time range
            fidelity_minutes: Data resolution (1, 60, 1440)

        Returns:
            List of price history points

        Notes:
            - The CLOB API requires interval='max' for historical data
            - fidelity parameter only works with explicit time ranges
            - Interval 'max' returns full history with auto-adjusted fidelity
        """
        # Always use interval=max to get full history - fidelity alone doesn't work
        params = {
            "market": market_id,
            "interval": "max",
        }

        try:
            response = await self.get("/prices-history", params=params)
            history = response.get("history", [])

            # Filter by time range if specified
            if start_time or end_time:
                filtered = []
                start_ts = int(start_time.timestamp()) if start_time else 0
                end_ts = int(end_time.timestamp()) if end_time else float('inf')
                for point in history:
                    t = point.get("t", 0)
                    if start_ts <= t <= end_ts:
                        filtered.append(point)
                history = filtered

            logger.debug(f"Fetched {len(history)} price points for {market_id}")
            return history

        except Exception as e:
            logger.warning(f"Failed to fetch price history for {market_id}: {e}")
            return []

    async def fetch_orderbook(
        self,
        token_id: str,
    ) -> Dict[str, Any]:
        """
        Fetch current orderbook for spread analysis.

        Args:
            token_id: Token ID for the outcome

        Returns:
            Orderbook with bids and asks
        """
        try:
            return await self.get("/book", params={"token_id": token_id})
        except Exception as e:
            logger.warning(f"Failed to fetch orderbook for {token_id}: {e}")
            return {}

    async def fetch_market_trades(
        self,
        token_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent trades for a market.

        Args:
            token_id: Token ID for the outcome
            limit: Maximum trades to fetch

        Returns:
            List of trade records
        """
        try:
            response = await self.get(
                "/trades",
                params={"asset_id": token_id, "limit": limit},
            )
            return response.get("trades", [])
        except Exception as e:
            logger.warning(f"Failed to fetch trades for {token_id}: {e}")
            return []

    def normalize_market(self, raw: Dict[str, Any]) -> MarketMetadata:
        """
        Convert Gamma API market response to unified schema.

        Args:
            raw: Raw market data from Gamma API

        Returns:
            Normalized MarketMetadata object
        """
        # Extract outcomes and tokens
        outcomes = raw.get("outcomes", [])
        if isinstance(outcomes, str):
            import json
            try:
                outcomes = json.loads(outcomes)
            except json.JSONDecodeError:
                outcomes = []

        clob_tokens = raw.get("clobTokenIds", [])
        if isinstance(clob_tokens, str):
            import json
            try:
                clob_tokens = json.loads(clob_tokens)
            except json.JSONDecodeError:
                clob_tokens = []

        # Determine if binary (any 2-outcome market is considered binary)
        # Common patterns: Yes/No, Up/Down, Over/Under, etc.
        is_binary = len(outcomes) == 2

        # Get token IDs (first outcome token)
        yes_token = clob_tokens[0] if len(clob_tokens) >= 1 else None
        no_token = clob_tokens[1] if len(clob_tokens) >= 2 else None

        # For Yes/No markets, ensure Yes token is first
        if len(outcomes) >= 2:
            outcomes_lower = [o.lower() for o in outcomes]
            if outcomes_lower[0] == "no" and outcomes_lower[1] == "yes":
                yes_token, no_token = no_token, yes_token

        # Build market ID with platform prefix
        condition_id = raw.get("conditionId", raw.get("id", ""))
        market_id = f"poly_{condition_id}"

        # Extract category from tags or dedicated field
        tags = raw.get("tags", [])
        category = raw.get("category", "")
        if not category and tags:
            category = tags[0]

        return MarketMetadata(
            market_id=market_id,
            platform=Platform.POLYMARKET,
            slug=raw.get("slug", ""),
            question=raw.get("question", ""),
            description=raw.get("description"),
            category=category or "uncategorized",
            series_ticker=raw.get("groupItemTitle") or raw.get("eventSlug"),
            tags=tags,
            created_at=parse_timestamp(raw.get("createdAt")),
            open_time=parse_timestamp(raw.get("startDate")),
            close_time=parse_timestamp(raw.get("endDate")),
            resolution_time=parse_timestamp(raw.get("resolutionDate")),
            is_binary=is_binary,
            outcome_yes_id=yes_token,
            outcome_no_id=no_token,
            status=map_polymarket_status(raw),
            result=self._extract_result(raw),
            min_tick_size=raw.get("orderPriceMinTickSize"),
            volume_total=self._parse_float(raw.get("volumeNum")),
            liquidity=self._parse_float(raw.get("liquidityNum")),
            source_hash=MarketMetadata.compute_hash(raw),
        )

    def normalize_price_point(
        self,
        raw: Dict[str, Any],
        market_id: str,
    ) -> PricePoint:
        """
        Convert CLOB prices-history response to unified schema.

        Args:
            raw: Raw price point from API
            market_id: Parent market ID

        Returns:
            Normalized PricePoint object

        Notes:
            - CLOB prices-history returns: t (timestamp), p (price)
            - Some endpoints include OHLC data
        """
        # Parse timestamp (Unix seconds)
        timestamp = datetime.utcfromtimestamp(raw["t"])

        # Get price (already 0-1 from this endpoint)
        price = raw.get("p", 0.0)
        price = normalize_price(price, "polymarket")

        # Extract OHLC if available
        ohlc = {}
        if "o" in raw:
            ohlc["open"] = normalize_price(raw["o"], "polymarket")
        if "h" in raw:
            ohlc["high"] = normalize_price(raw["h"], "polymarket")
        if "l" in raw:
            ohlc["low"] = normalize_price(raw["l"], "polymarket")
        if "c" in raw:
            ohlc["close"] = normalize_price(raw["c"], "polymarket")

        return PricePoint(
            timestamp=timestamp,
            market_id=market_id,
            platform=Platform.POLYMARKET,
            price_yes=price,
            price_no=1.0 - price,
            # CLOB prices-history doesn't include bid/ask
            bid_yes=None,
            ask_yes=None,
            bid_no=None,
            ask_no=None,
            midpoint=price,
            spread=None,
            volume=self._parse_float(raw.get("v")),
            volume_usd=None,
            open_interest=None,
            liquidity_usd=None,
            open=ohlc.get("open"),
            high=ohlc.get("high"),
            low=ohlc.get("low"),
            close=ohlc.get("close"),
            fidelity_minutes=60,
            source="api",
        )

    def _extract_result(self, raw: Dict[str, Any]) -> Optional[str]:
        """Extract resolution result from market data."""
        # Check for explicit result
        result = raw.get("result")
        if result:
            return result.lower()

        # Check resolved outcome prices
        outcome_prices = raw.get("outcomePrices", [])
        if len(outcome_prices) >= 2:
            outcomes = raw.get("outcomes", ["Yes", "No"])
            try:
                prices = [float(p) for p in outcome_prices]
                if prices[0] > 0.99:
                    return outcomes[0].lower()
                elif prices[1] > 0.99:
                    return outcomes[1].lower()
            except (ValueError, IndexError):
                pass

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

    async def get_market_details(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information for a specific market.

        Args:
            condition_id: Market condition ID

        Returns:
            Market details or None
        """
        url = f"{self.gamma_url}/markets/{condition_id}"
        session = await self._ensure_session()

        try:
            await self._rate_limiter.acquire()
            async with session.get(url) as response:
                if response.status == 404:
                    return None
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch market details for {condition_id}: {e}")
            return None

    async def get_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch events (groups of related markets).

        Args:
            limit: Maximum events to fetch

        Returns:
            List of event objects
        """
        url = f"{self.gamma_url}/events"
        session = await self._ensure_session()

        try:
            await self._rate_limiter.acquire()
            async with session.get(url, params={"limit": limit}) as response:
                response.raise_for_status()
                data = await response.json()
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Failed to fetch events: {e}")
            return []
