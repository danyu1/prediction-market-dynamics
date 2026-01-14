"""
Data Ingestion Pipeline for Prediction Markets

This module orchestrates the data ingestion process, coordinating
between API clients and storage to fetch, normalize, and persist
prediction market data.

Features:
    - Full sync: Fetch all markets and complete price history
    - Incremental sync: Only fetch new data since last run
    - Multi-platform support: Polymarket and Kalshi
    - Market filtering: Binary markets with known resolution dates
    - Progress tracking and logging
    - CLI interface for manual runs

Usage:
    # Full sync (all data)
    python -m data.ingest --mode full

    # Incremental sync (new data only)
    python -m data.ingest --mode incremental

    # Specific platform
    python -m data.ingest --platforms polymarket

    # Limit markets (for testing)
    python -m data.ingest --max-markets 50
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import List, Optional

from tqdm import tqdm

from data.schema import MarketMetadata, Platform
from data.storage import ParquetStorage
from data.polymarket import PolymarketClient
from data.kalshi import KalshiClient
from data.base_client import BaseMarketClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class DataIngester:
    """
    Orchestrates data ingestion from multiple prediction market platforms.

    Coordinates fetching market metadata and price history from Polymarket
    and Kalshi, normalizing the data, and storing it in Parquet format.

    Attributes:
        storage: ParquetStorage instance for data persistence
        polymarket: PolymarketClient for Polymarket API
        kalshi: KalshiClient for Kalshi API
    """

    def __init__(
        self,
        storage_path: str = "data/store",
        min_price_points: int = 5,
        max_markets_per_platform: Optional[int] = None,
    ):
        """
        Initialize the data ingester.

        Args:
            storage_path: Path for data storage
            min_price_points: Minimum price points required per market
            max_markets_per_platform: Maximum markets to process (for testing)
        """
        self.storage = ParquetStorage(storage_path)
        self.min_price_points = min_price_points
        self.max_markets = max_markets_per_platform

        # Clients will be initialized in context manager
        self.polymarket: Optional[PolymarketClient] = None
        self.kalshi: Optional[KalshiClient] = None

    async def __aenter__(self):
        """Initialize API clients."""
        self.polymarket = PolymarketClient()
        self.kalshi = KalshiClient()
        await self.polymarket.__aenter__()
        await self.kalshi.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close API clients."""
        if self.polymarket:
            await self.polymarket.__aexit__(exc_type, exc_val, exc_tb)
        if self.kalshi:
            await self.kalshi.__aexit__(exc_type, exc_val, exc_tb)

    def _get_client(self, platform: Platform) -> BaseMarketClient:
        """Get the appropriate client for a platform."""
        if platform == Platform.POLYMARKET:
            return self.polymarket
        elif platform == Platform.KALSHI:
            return self.kalshi
        else:
            raise ValueError(f"Unknown platform: {platform}")

    def _passes_filters(self, market: MarketMetadata) -> bool:
        """
        Check if a market passes ingestion filters.

        Filters:
            - Binary outcome only
            - Has outcome token ID
            - (Relaxed: close_time not strictly required)

        Args:
            market: Market to check

        Returns:
            True if market passes all filters
        """
        # Must be binary
        if not market.is_binary:
            return False

        # Must have token ID for price fetching
        if not market.outcome_yes_id:
            return False

        # Note: close_time filter relaxed - many active markets don't have it set
        return True

    async def run_full_sync(
        self,
        platforms: Optional[List[Platform]] = None,
    ) -> dict:
        """
        Perform full sync of all markets and price history.

        Fetches all markets matching filters and their complete
        price history from the specified platforms.

        Args:
            platforms: Platforms to sync (default: all)

        Returns:
            Dictionary with sync statistics
        """
        platforms = platforms or [Platform.POLYMARKET, Platform.KALSHI]

        stats = {
            "start_time": datetime.utcnow(),
            "platforms": {},
            "total_markets": 0,
            "total_price_points": 0,
        }

        for platform in platforms:
            logger.info(f"Starting full sync for {platform.value}")
            platform_stats = await self._sync_platform(platform, incremental=False)
            stats["platforms"][platform.value] = platform_stats
            stats["total_markets"] += platform_stats["markets_processed"]
            stats["total_price_points"] += platform_stats["price_points"]

        stats["end_time"] = datetime.utcnow()
        stats["duration_seconds"] = (
            stats["end_time"] - stats["start_time"]
        ).total_seconds()

        logger.info(
            f"Full sync complete: {stats['total_markets']} markets, "
            f"{stats['total_price_points']} price points in "
            f"{stats['duration_seconds']:.1f}s"
        )

        return stats

    async def run_incremental_sync(
        self,
        platforms: Optional[List[Platform]] = None,
    ) -> dict:
        """
        Perform incremental sync (new data only).

        Only fetches new price data since the last sync for each market.
        New markets are fully synced.

        Args:
            platforms: Platforms to sync (default: all)

        Returns:
            Dictionary with sync statistics
        """
        platforms = platforms or [Platform.POLYMARKET, Platform.KALSHI]

        stats = {
            "start_time": datetime.utcnow(),
            "platforms": {},
            "total_markets": 0,
            "total_price_points": 0,
        }

        for platform in platforms:
            logger.info(f"Starting incremental sync for {platform.value}")
            platform_stats = await self._sync_platform(platform, incremental=True)
            stats["platforms"][platform.value] = platform_stats
            stats["total_markets"] += platform_stats["markets_processed"]
            stats["total_price_points"] += platform_stats["price_points"]

        stats["end_time"] = datetime.utcnow()
        stats["duration_seconds"] = (
            stats["end_time"] - stats["start_time"]
        ).total_seconds()

        logger.info(
            f"Incremental sync complete: {stats['total_markets']} markets, "
            f"{stats['total_price_points']} new price points in "
            f"{stats['duration_seconds']:.1f}s"
        )

        return stats

    async def _sync_platform(
        self,
        platform: Platform,
        incremental: bool = False,
    ) -> dict:
        """
        Sync a single platform.

        Args:
            platform: Platform to sync
            incremental: If True, only fetch new data

        Returns:
            Platform sync statistics
        """
        client = self._get_client(platform)

        stats = {
            "markets_fetched": 0,
            "markets_filtered": 0,
            "markets_processed": 0,
            "price_points": 0,
            "errors": 0,
        }

        # Fetch all markets
        logger.info(f"Fetching markets from {platform.value}...")
        markets = []

        # Use specialized iterators to find markets with volume
        if platform == Platform.KALSHI:
            market_iterator = self.kalshi.iterate_markets_with_volume(
                max_markets=self.max_markets,
            )
        elif platform == Platform.POLYMARKET:
            market_iterator = self.polymarket.iterate_markets_with_volume(
                max_markets=self.max_markets,
                min_volume=1000.0,  # Minimum $1000 volume
            )
        else:
            market_iterator = client.iterate_all_markets(
                max_markets=self.max_markets,
            )

        async for market in market_iterator:
            stats["markets_fetched"] += 1

            if self._passes_filters(market):
                markets.append(market)
            else:
                stats["markets_filtered"] += 1

        logger.info(
            f"Found {len(markets)} markets after filtering "
            f"(filtered out {stats['markets_filtered']})"
        )

        # Save market metadata
        if markets:
            self.storage.write_markets(markets, platform)

        # Fetch price history for each market
        logger.info(f"Fetching price history for {len(markets)} markets...")

        for market in tqdm(markets, desc=f"{platform.value} prices"):
            try:
                # Determine start time for incremental sync
                start_time = None
                if incremental:
                    last_time = self.storage.get_last_price_time(
                        platform, market.market_id
                    )
                    if last_time:
                        # Add small buffer to avoid duplicates
                        start_time = last_time - timedelta(hours=1)

                # Fetch and normalize prices
                prices = await client.fetch_and_normalize_prices(
                    market,
                    start_time=start_time,
                    fidelity_minutes=60,
                )

                if len(prices) >= self.min_price_points:
                    self.storage.write_prices(prices, platform)
                    stats["price_points"] += len(prices)
                    stats["markets_processed"] += 1
                else:
                    logger.debug(
                        f"Skipping {market.market_id}: "
                        f"only {len(prices)} price points"
                    )

            except Exception as e:
                logger.warning(f"Error processing {market.market_id}: {e}")
                stats["errors"] += 1

        # Log client stats
        logger.info(f"{platform.value} API stats: {client.get_stats_summary()}")

        return stats

    def get_storage_summary(self) -> dict:
        """Get summary of stored data."""
        return self.storage.get_storage_stats()


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prediction Market Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m data.ingest --mode full
  python -m data.ingest --mode incremental --platforms polymarket
  python -m data.ingest --max-markets 100 --verbose
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help="Sync mode: full (all data) or incremental (new only)",
    )

    parser.add_argument(
        "--platforms",
        nargs="+",
        choices=["polymarket", "kalshi"],
        default=["polymarket", "kalshi"],
        help="Platforms to sync",
    )

    parser.add_argument(
        "--max-markets",
        type=int,
        default=None,
        help="Maximum markets per platform (for testing)",
    )

    parser.add_argument(
        "--storage-path",
        default="data/store",
        help="Path for data storage",
    )

    parser.add_argument(
        "--min-price-points",
        type=int,
        default=5,
        help="Minimum price points required per market",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show storage statistics, don't sync",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse platforms
    platforms = [Platform(p) for p in args.platforms]

    # Initialize ingester
    async with DataIngester(
        storage_path=args.storage_path,
        min_price_points=args.min_price_points,
        max_markets_per_platform=args.max_markets,
    ) as ingester:

        if args.stats_only:
            stats = ingester.get_storage_summary()
            print("\n=== Storage Statistics ===")
            print(f"Total size: {stats['total_size_mb']:.2f} MB")
            print("\nMarkets:")
            for platform, data in stats["markets"].items():
                print(f"  {platform}: {data['count']} markets ({data['size_mb']:.2f} MB)")
            print("\nPrices:")
            for platform, data in stats["prices"].items():
                print(
                    f"  {platform}: {data['rows']:,} rows, "
                    f"{data['partitions']} partitions ({data['size_mb']:.2f} MB)"
                )
            return

        # Run sync
        if args.mode == "full":
            stats = await ingester.run_full_sync(platforms)
        else:
            stats = await ingester.run_incremental_sync(platforms)

        # Print summary
        print("\n=== Ingestion Complete ===")
        print(f"Duration: {stats['duration_seconds']:.1f} seconds")
        print(f"Total markets: {stats['total_markets']}")
        print(f"Total price points: {stats['total_price_points']:,}")

        for platform, pstats in stats["platforms"].items():
            print(f"\n{platform}:")
            print(f"  Markets fetched: {pstats['markets_fetched']}")
            print(f"  Markets filtered: {pstats['markets_filtered']}")
            print(f"  Markets processed: {pstats['markets_processed']}")
            print(f"  Price points: {pstats['price_points']:,}")
            if pstats['errors'] > 0:
                print(f"  Errors: {pstats['errors']}")

        # Show storage stats
        storage_stats = ingester.get_storage_summary()
        print(f"\nTotal storage: {storage_stats['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    asyncio.run(main())
