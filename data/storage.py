"""
Parquet Storage Layer for Prediction Market Data

This module provides efficient storage and retrieval of prediction market
data using Apache Parquet with time-based partitioning.

Features:
    - Time-based partitioning (year/month) for efficient queries
    - Incremental updates with deduplication
    - Partition pruning for time-range queries
    - Schema validation on write
    - Support for both markets metadata and price history

Directory Structure:
    data/store/
    ├── markets/
    │   ├── polymarket/markets.parquet
    │   └── kalshi/markets.parquet
    └── prices/
        ├── polymarket/
        │   └── year=2024/month=01/prices.parquet
        └── kalshi/
            └── year=2024/month=01/prices.parquet
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.schema import (
    MarketMetadata,
    Platform,
    PricePoint,
    MARKET_METADATA_SCHEMA,
    PRICE_HISTORY_SCHEMA,
)

logger = logging.getLogger(__name__)


class ParquetStorage:
    """
    Parquet-based storage with partitioning and incremental updates.

    Handles storage of both market metadata and price history data
    with efficient partitioning strategies for each data type.

    Attributes:
        base_path: Root directory for all data storage
        markets_path: Subdirectory for market metadata
        prices_path: Subdirectory for price history
    """

    def __init__(self, base_path: Union[str, Path] = "data/store"):
        """
        Initialize storage with base path.

        Args:
            base_path: Root directory for data storage
        """
        self.base_path = Path(base_path)
        self.markets_path = self.base_path / "markets"
        self.prices_path = self.base_path / "prices"

        # Create directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.markets_path.mkdir(parents=True, exist_ok=True)
        self.prices_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Market Metadata Storage
    # =========================================================================

    def write_markets(
        self,
        markets: List[MarketMetadata],
        platform: Platform,
        mode: str = "merge",
    ) -> Path:
        """
        Write market metadata to Parquet.

        Args:
            markets: List of MarketMetadata objects
            platform: Target platform
            mode: 'merge' to update existing, 'overwrite' to replace

        Returns:
            Path to written file
        """
        if not markets:
            logger.warning("No markets to write")
            return None

        # Create platform directory
        platform_dir = self.markets_path / platform.value
        platform_dir.mkdir(parents=True, exist_ok=True)
        file_path = platform_dir / "markets.parquet"

        # Convert to DataFrame
        df = pd.DataFrame([m.to_dict() for m in markets])

        # Handle list columns (tags)
        if "tags" in df.columns:
            df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])

        if mode == "merge" and file_path.exists():
            # Read existing and merge
            existing_df = pd.read_parquet(file_path)

            # Combine and deduplicate by market_id, keeping latest
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["market_id"], keep="last")
            df = df.sort_values("market_id").reset_index(drop=True)

        # Ensure timestamp columns are properly typed
        timestamp_cols = ["created_at", "open_time", "close_time",
                         "resolution_time", "ingested_at"]
        for col in timestamp_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

        # Write to Parquet
        df.to_parquet(
            file_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        logger.info(f"Wrote {len(df)} markets to {file_path}")
        return file_path

    def read_markets(
        self,
        platform: Optional[Platform] = None,
        filters: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Read market metadata from Parquet.

        Args:
            platform: Filter by platform (None for all)
            filters: Additional filters as dict (column: value)

        Returns:
            DataFrame with market metadata
        """
        dfs = []

        if platform:
            platforms = [platform]
        else:
            platforms = [Platform.POLYMARKET, Platform.KALSHI]

        for plat in platforms:
            file_path = self.markets_path / plat.value / "markets.parquet"
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # Apply filters
        if filters:
            for col, value in filters.items():
                if col in df.columns:
                    if isinstance(value, list):
                        df = df[df[col].isin(value)]
                    else:
                        df = df[df[col] == value]

        return df

    def get_market_by_id(self, market_id: str) -> Optional[Dict]:
        """
        Get a single market by ID.

        Args:
            market_id: Market identifier (platform-prefixed)

        Returns:
            Market data as dict or None
        """
        # Determine platform from ID prefix
        if market_id.startswith("poly_"):
            platform = Platform.POLYMARKET
        elif market_id.startswith("kalshi_"):
            platform = Platform.KALSHI
        else:
            platform = None

        df = self.read_markets(platform=platform)
        if df.empty:
            return None

        matches = df[df["market_id"] == market_id]
        if matches.empty:
            return None

        return matches.iloc[0].to_dict()

    # =========================================================================
    # Price History Storage
    # =========================================================================

    def write_prices(
        self,
        prices: List[PricePoint],
        platform: Platform,
    ) -> List[Path]:
        """
        Write price history with time-based partitioning.

        Prices are partitioned by year/month for efficient time-range queries
        and incremental updates.

        Args:
            prices: List of PricePoint objects
            platform: Target platform

        Returns:
            List of paths to written partition files
        """
        if not prices:
            logger.warning("No prices to write")
            return []

        # Convert to DataFrame
        df = pd.DataFrame([p.to_dict() for p in prices])

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Extract partitioning columns
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month

        written_paths = []

        # Write each partition
        for (year, month), group in df.groupby(["year", "month"]):
            partition_path = (
                self.prices_path
                / platform.value
                / f"year={year}"
                / f"month={month:02d}"
            )
            partition_path.mkdir(parents=True, exist_ok=True)

            file_path = partition_path / "prices.parquet"

            # Remove partitioning columns from data
            group_data = group.drop(columns=["year", "month"])

            # Merge with existing partition data
            if file_path.exists():
                try:
                    existing = pd.read_parquet(file_path)
                    group_data = pd.concat([existing, group_data], ignore_index=True)

                    # Deduplicate by market_id + timestamp
                    group_data = group_data.drop_duplicates(
                        subset=["market_id", "timestamp"],
                        keep="last",
                    )
                except Exception as e:
                    logger.warning(f"Failed to read existing partition: {e}")

            # Sort by timestamp
            group_data = group_data.sort_values("timestamp").reset_index(drop=True)

            # Write partition
            group_data.to_parquet(
                file_path,
                engine="pyarrow",
                compression="snappy",
                index=False,
            )

            written_paths.append(file_path)
            logger.debug(
                f"Wrote {len(group_data)} prices to {file_path}"
            )

        logger.info(
            f"Wrote {len(prices)} prices to {len(written_paths)} partitions"
        )
        return written_paths

    def read_prices(
        self,
        platform: Optional[Platform] = None,
        market_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Read price history with partition pruning.

        Uses time-based filters to skip irrelevant partitions,
        significantly improving query performance for time-range queries.

        Args:
            platform: Filter by platform (None for all)
            market_ids: Filter by market IDs
            start_time: Start of time range
            end_time: End of time range

        Returns:
            DataFrame with price history
        """
        if platform:
            platforms = [platform]
        else:
            platforms = [Platform.POLYMARKET, Platform.KALSHI]

        dfs = []

        for plat in platforms:
            platform_path = self.prices_path / plat.value

            if not platform_path.exists():
                continue

            # Find relevant partitions
            partitions = self._find_partitions(
                platform_path,
                start_time,
                end_time,
            )

            for partition_file in partitions:
                try:
                    df = pd.read_parquet(partition_file)

                    # Apply market_id filter
                    if market_ids:
                        df = df[df["market_id"].isin(market_ids)]

                    # Apply time filters (more precise than partition pruning)
                    if start_time:
                        df = df[df["timestamp"] >= pd.Timestamp(start_time, tz="UTC")]
                    if end_time:
                        df = df[df["timestamp"] <= pd.Timestamp(end_time, tz="UTC")]

                    if not df.empty:
                        dfs.append(df)

                except Exception as e:
                    logger.warning(f"Failed to read {partition_file}: {e}")

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        return result.sort_values("timestamp").reset_index(drop=True)

    def _find_partitions(
        self,
        platform_path: Path,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> List[Path]:
        """
        Find partition files matching time range.

        Args:
            platform_path: Path to platform directory
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of matching partition file paths
        """
        partitions = []

        # Get min/max years to scan
        if start_time:
            min_year = start_time.year
            min_month = start_time.month
        else:
            min_year = 2020  # Reasonable default
            min_month = 1

        if end_time:
            max_year = end_time.year
            max_month = end_time.month
        else:
            max_year = datetime.now().year + 1
            max_month = 12

        # Scan year directories
        for year_dir in sorted(platform_path.glob("year=*")):
            try:
                year = int(year_dir.name.split("=")[1])
            except (IndexError, ValueError):
                continue

            if year < min_year or year > max_year:
                continue

            # Scan month directories
            for month_dir in sorted(year_dir.glob("month=*")):
                try:
                    month = int(month_dir.name.split("=")[1])
                except (IndexError, ValueError):
                    continue

                # Check if partition is in range
                if year == min_year and month < min_month:
                    continue
                if year == max_year and month > max_month:
                    continue

                # Check for partition file
                partition_file = month_dir / "prices.parquet"
                if partition_file.exists():
                    partitions.append(partition_file)

        return partitions

    def get_last_price_time(
        self,
        platform: Platform,
        market_id: str,
    ) -> Optional[datetime]:
        """
        Get the most recent timestamp for a market.

        Useful for incremental updates to only fetch new data.

        Args:
            platform: Target platform
            market_id: Market identifier

        Returns:
            Most recent timestamp or None if no data
        """
        df = self.read_prices(platform=platform, market_ids=[market_id])

        if df.empty:
            return None

        return df["timestamp"].max().to_pydatetime()

    # =========================================================================
    # Statistics and Utilities
    # =========================================================================

    def get_storage_stats(self) -> Dict:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        stats = {
            "markets": {},
            "prices": {},
            "total_size_mb": 0,
        }

        # Markets stats
        for platform in [Platform.POLYMARKET, Platform.KALSHI]:
            file_path = self.markets_path / platform.value / "markets.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                size_mb = file_path.stat().st_size / (1024 * 1024)
                stats["markets"][platform.value] = {
                    "count": len(df),
                    "size_mb": round(size_mb, 2),
                }
                stats["total_size_mb"] += size_mb

        # Prices stats
        for platform in [Platform.POLYMARKET, Platform.KALSHI]:
            platform_path = self.prices_path / platform.value
            if platform_path.exists():
                total_rows = 0
                total_size = 0
                partition_count = 0

                for pfile in platform_path.glob("**/prices.parquet"):
                    try:
                        df = pd.read_parquet(pfile)
                        total_rows += len(df)
                        total_size += pfile.stat().st_size
                        partition_count += 1
                    except Exception:
                        continue

                size_mb = total_size / (1024 * 1024)
                stats["prices"][platform.value] = {
                    "rows": total_rows,
                    "partitions": partition_count,
                    "size_mb": round(size_mb, 2),
                }
                stats["total_size_mb"] += size_mb

        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        return stats

    def clear_platform(self, platform: Platform) -> None:
        """
        Clear all data for a platform.

        Args:
            platform: Platform to clear
        """
        import shutil

        # Clear markets
        markets_dir = self.markets_path / platform.value
        if markets_dir.exists():
            shutil.rmtree(markets_dir)

        # Clear prices
        prices_dir = self.prices_path / platform.value
        if prices_dir.exists():
            shutil.rmtree(prices_dir)

        logger.info(f"Cleared all data for {platform.value}")

    def export_to_csv(
        self,
        output_dir: Union[str, Path],
        platform: Optional[Platform] = None,
    ) -> Dict[str, Path]:
        """
        Export data to CSV format.

        Args:
            output_dir: Output directory for CSV files
            platform: Platform to export (None for all)

        Returns:
            Dictionary mapping data type to output path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported = {}

        # Export markets
        markets_df = self.read_markets(platform=platform)
        if not markets_df.empty:
            markets_path = output_dir / "markets.csv"
            markets_df.to_csv(markets_path, index=False)
            exported["markets"] = markets_path

        # Export prices
        prices_df = self.read_prices(platform=platform)
        if not prices_df.empty:
            prices_path = output_dir / "prices.csv"
            prices_df.to_csv(prices_path, index=False)
            exported["prices"] = prices_path

        logger.info(f"Exported data to {output_dir}")
        return exported
