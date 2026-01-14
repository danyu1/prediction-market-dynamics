"""
Data Schema Definitions for Prediction Market Analysis

This module defines the unified data models and Parquet schemas for
storing prediction market data from multiple platforms (Polymarket, Kalshi).

Schema Design Principles:
    - All prices normalized to [0, 1] probability scale
    - All timestamps in UTC
    - Platform-prefixed IDs to prevent collision
    - Consistent naming across platforms
    - Optional fields for platform-specific data

Parquet Schema Notes:
    - Uses PyArrow for schema definition
    - Snappy compression for balance of speed and size
    - Time-based partitioning (year/month) for efficient queries
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional
import hashlib
import json

import pyarrow as pa


class Platform(Enum):
    """Supported prediction market platforms."""
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class MarketStatus(Enum):
    """Unified market status across platforms.

    Mapping:
        Polymarket: active -> OPEN, closed -> CLOSED, resolved -> SETTLED
        Kalshi: unopened -> UNOPENED, open -> OPEN, paused -> PAUSED,
                closed -> CLOSED, settled -> SETTLED
    """
    UNOPENED = "unopened"
    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"
    SETTLED = "settled"


@dataclass
class MarketMetadata:
    """
    Unified market metadata schema for cross-platform analysis.

    This dataclass represents the normalized metadata for a single
    prediction market, regardless of source platform.

    Attributes:
        market_id: Platform-prefixed unique identifier (e.g., 'poly_abc123')
        platform: Source platform (POLYMARKET or KALSHI)
        slug: Human-readable URL slug
        question: Full market question text
        description: Detailed market description (if available)
        category: Primary category for grouping
        series_ticker: Parent series identifier (Kalshi) or event slug (Polymarket)
        tags: Additional classification tags
        created_at: When the market was created (UTC)
        open_time: When trading opened (UTC)
        close_time: Expected/actual close time (UTC)
        resolution_time: When market was resolved (UTC, None if unresolved)
        is_binary: True for Yes/No markets only
        outcome_yes_id: Token/contract ID for Yes outcome
        outcome_no_id: Token/contract ID for No outcome (if separate)
        status: Current market status
        result: Resolution result ("yes", "no", or None)
        min_tick_size: Minimum price increment
        volume_total: Total volume traded (if available)
        liquidity: Current liquidity (if available)
        ingested_at: When this record was pulled (UTC)
        source_hash: MD5 hash of raw response for deduplication
    """
    market_id: str
    platform: Platform
    slug: str
    question: str
    description: Optional[str] = None
    category: str = "unknown"
    series_ticker: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    resolution_time: Optional[datetime] = None
    is_binary: bool = True
    outcome_yes_id: Optional[str] = None
    outcome_no_id: Optional[str] = None
    status: MarketStatus = MarketStatus.OPEN
    result: Optional[str] = None
    min_tick_size: Optional[float] = None
    volume_total: Optional[float] = None
    liquidity: Optional[float] = None
    ingested_at: datetime = field(default_factory=datetime.utcnow)
    source_hash: str = ""

    def __post_init__(self):
        """Ensure platform enum is properly set."""
        if isinstance(self.platform, str):
            self.platform = Platform(self.platform)
        if isinstance(self.status, str):
            self.status = MarketStatus(self.status)

    @staticmethod
    def compute_hash(raw_data: dict) -> str:
        """Compute MD5 hash of raw API response for deduplication."""
        return hashlib.md5(
            json.dumps(raw_data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "market_id": self.market_id,
            "platform": self.platform.value,
            "slug": self.slug,
            "question": self.question,
            "description": self.description,
            "category": self.category,
            "series_ticker": self.series_ticker,
            "tags": self.tags,
            "created_at": self.created_at,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "resolution_time": self.resolution_time,
            "is_binary": self.is_binary,
            "outcome_yes_id": self.outcome_yes_id,
            "outcome_no_id": self.outcome_no_id,
            "status": self.status.value,
            "result": self.result,
            "min_tick_size": self.min_tick_size,
            "volume_total": self.volume_total,
            "liquidity": self.liquidity,
            "ingested_at": self.ingested_at,
            "source_hash": self.source_hash,
        }


@dataclass
class PricePoint:
    """
    Individual price observation for time series data.

    Represents a single data point in a market's price history,
    normalized across platforms.

    Attributes:
        timestamp: Observation time (UTC)
        market_id: Reference to parent market
        platform: Source platform
        price_yes: Probability of Yes outcome [0, 1]
        price_no: Probability of No outcome [0, 1] (1 - price_yes for binary)
        bid_yes: Best bid price for Yes (if available)
        ask_yes: Best ask price for Yes (if available)
        bid_no: Best bid price for No (if available)
        ask_no: Best ask price for No (if available)
        midpoint: Mid-market price (bid + ask) / 2
        spread: Bid-ask spread (ask - bid)
        volume: Contracts/shares traded in this period
        volume_usd: Dollar volume (if available)
        open_interest: Outstanding contracts (if available)
        liquidity_usd: Available liquidity in USD (if available)
        open: Opening price (OHLC)
        high: High price (OHLC)
        low: Low price (OHLC)
        close: Closing price (OHLC)
        fidelity_minutes: Time resolution of this data point
        source: Data source indicator ('api', 'derived', 'interpolated')
    """
    timestamp: datetime
    market_id: str
    platform: Platform
    price_yes: float
    price_no: Optional[float] = None
    bid_yes: Optional[float] = None
    ask_yes: Optional[float] = None
    bid_no: Optional[float] = None
    ask_no: Optional[float] = None
    midpoint: Optional[float] = None
    spread: Optional[float] = None
    volume: Optional[float] = None
    volume_usd: Optional[float] = None
    open_interest: Optional[float] = None
    liquidity_usd: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    fidelity_minutes: int = 60
    source: str = "api"

    def __post_init__(self):
        """Compute derived fields and validate."""
        if isinstance(self.platform, str):
            self.platform = Platform(self.platform)

        # Compute price_no if not provided
        if self.price_no is None:
            self.price_no = 1.0 - self.price_yes

        # Compute midpoint if bid/ask available
        if self.midpoint is None and self.bid_yes is not None and self.ask_yes is not None:
            self.midpoint = (self.bid_yes + self.ask_yes) / 2

        # Compute spread if bid/ask available
        if self.spread is None and self.bid_yes is not None and self.ask_yes is not None:
            self.spread = self.ask_yes - self.bid_yes

        # Validate probability bounds
        self._validate_probability(self.price_yes, "price_yes")
        self._validate_probability(self.price_no, "price_no")

    @staticmethod
    def _validate_probability(value: Optional[float], name: str) -> None:
        """Ensure probability is in [0, 1] range."""
        if value is not None and (value < 0 or value > 1):
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "timestamp": self.timestamp,
            "market_id": self.market_id,
            "platform": self.platform.value,
            "price_yes": self.price_yes,
            "price_no": self.price_no,
            "bid_yes": self.bid_yes,
            "ask_yes": self.ask_yes,
            "bid_no": self.bid_no,
            "ask_no": self.ask_no,
            "midpoint": self.midpoint,
            "spread": self.spread,
            "volume": self.volume,
            "volume_usd": self.volume_usd,
            "open_interest": self.open_interest,
            "liquidity_usd": self.liquidity_usd,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "fidelity_minutes": self.fidelity_minutes,
            "source": self.source,
        }


# =============================================================================
# PyArrow Schemas for Parquet Storage
# =============================================================================

MARKET_METADATA_SCHEMA = pa.schema([
    ("market_id", pa.string()),
    ("platform", pa.string()),
    ("slug", pa.string()),
    ("question", pa.string()),
    ("description", pa.string()),
    ("category", pa.string()),
    ("series_ticker", pa.string()),
    ("tags", pa.list_(pa.string())),
    ("created_at", pa.timestamp("us", tz="UTC")),
    ("open_time", pa.timestamp("us", tz="UTC")),
    ("close_time", pa.timestamp("us", tz="UTC")),
    ("resolution_time", pa.timestamp("us", tz="UTC")),
    ("is_binary", pa.bool_()),
    ("outcome_yes_id", pa.string()),
    ("outcome_no_id", pa.string()),
    ("status", pa.string()),
    ("result", pa.string()),
    ("min_tick_size", pa.float64()),
    ("volume_total", pa.float64()),
    ("liquidity", pa.float64()),
    ("ingested_at", pa.timestamp("us", tz="UTC")),
    ("source_hash", pa.string()),
])

PRICE_HISTORY_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("market_id", pa.string()),
    ("platform", pa.string()),
    ("price_yes", pa.float64()),
    ("price_no", pa.float64()),
    ("bid_yes", pa.float64()),
    ("ask_yes", pa.float64()),
    ("bid_no", pa.float64()),
    ("ask_no", pa.float64()),
    ("midpoint", pa.float64()),
    ("spread", pa.float64()),
    ("volume", pa.float64()),
    ("volume_usd", pa.float64()),
    ("open_interest", pa.float64()),
    ("liquidity_usd", pa.float64()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("fidelity_minutes", pa.int32()),
    ("source", pa.string()),
])


# =============================================================================
# Normalization Utilities
# =============================================================================

def normalize_price(price: float, source: str = "unknown") -> float:
    """
    Normalize price to [0, 1] probability scale.

    Args:
        price: Raw price value
        source: Platform source for context

    Returns:
        Normalized probability in [0, 1]

    Notes:
        - Polymarket: Usually already 0-1, but some endpoints return 0-100
        - Kalshi: Returns cents (0-100), divide by 100
    """
    if price > 1:
        # Likely percentage or cents format
        return price / 100.0
    return price


def parse_timestamp(value, default: Optional[datetime] = None) -> Optional[datetime]:
    """
    Parse various timestamp formats to datetime.

    Args:
        value: Timestamp in various formats (Unix seconds, ISO string, etc.)
        default: Default value if parsing fails

    Returns:
        datetime object in UTC or default
    """
    if value is None:
        return default

    if isinstance(value, datetime):
        return value

    if isinstance(value, (int, float)):
        # Unix timestamp (seconds)
        try:
            return datetime.utcfromtimestamp(value)
        except (ValueError, OSError):
            return default

    if isinstance(value, str):
        # ISO format or similar
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

        # Try pandas for more flexible parsing
        try:
            import pandas as pd
            return pd.to_datetime(value).to_pydatetime()
        except Exception:
            pass

    return default


def map_polymarket_status(raw: dict) -> MarketStatus:
    """
    Map Polymarket status fields to unified MarketStatus.

    Polymarket uses multiple fields:
        - active: bool
        - closed: bool
        - archived: bool
    """
    if raw.get("archived", False):
        return MarketStatus.SETTLED
    if raw.get("closed", False):
        return MarketStatus.CLOSED
    if raw.get("active", True):
        return MarketStatus.OPEN
    return MarketStatus.UNOPENED


def map_kalshi_status(status: str) -> MarketStatus:
    """
    Map Kalshi status string to unified MarketStatus.

    Kalshi statuses: unopened, open, paused, closed, settled
    """
    status_map = {
        "unopened": MarketStatus.UNOPENED,
        "open": MarketStatus.OPEN,
        "paused": MarketStatus.PAUSED,
        "closed": MarketStatus.CLOSED,
        "settled": MarketStatus.SETTLED,
    }
    return status_map.get(status.lower(), MarketStatus.OPEN)
