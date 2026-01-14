"""
Prediction Market Data Ingestion Package

This package provides tools for fetching, normalizing, and analyzing
prediction market data from Polymarket and Kalshi.

Modules:
    schema: Data models and Parquet schemas
    base_client: Abstract API client with rate limiting
    polymarket: Polymarket CLOB + Gamma API client
    kalshi: Kalshi Trade API v2 client
    storage: Parquet storage with time partitioning
    ingest: Main orchestration and CLI
    eda: Exploratory data analysis and visualization
"""

from data.schema import Platform, MarketStatus, MarketMetadata, PricePoint

__all__ = [
    "Platform",
    "MarketStatus",
    "MarketMetadata",
    "PricePoint",
]
