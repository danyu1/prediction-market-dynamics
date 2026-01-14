"""
Exploratory Data Analysis for Prediction Markets

This module provides comprehensive analysis and visualization of
prediction market data from Polymarket and Kalshi.

Features:
    - Price trajectory visualizations
    - Return distribution analysis
    - Volatility and market quality metrics
    - Cross-platform comparison
    - Volume and liquidity patterns

Usage:
    python -m data.eda --output-dir output/plots
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from data.storage import ParquetStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class PredictionMarketEDA:
    """
    Exploratory Data Analysis for prediction market data.

    Loads data from Parquet storage and generates various
    visualizations and statistical analyses.
    """

    def __init__(
        self,
        storage_path: str = "data/store",
        output_dir: str = "output/plots",
    ):
        """
        Initialize EDA analyzer.

        Args:
            storage_path: Path to Parquet data storage
            output_dir: Directory for output plots
        """
        self.storage = ParquetStorage(storage_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.prices_poly: Optional[pd.DataFrame] = None
        self.prices_kalshi: Optional[pd.DataFrame] = None
        self.markets_poly: Optional[pd.DataFrame] = None
        self.markets_kalshi: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """Load all data from storage."""
        logger.info("Loading data from storage...")

        try:
            self.prices_poly = pd.read_parquet(
                f"{self.storage.base_path}/prices/polymarket/"
            )
            logger.info(f"Loaded {len(self.prices_poly):,} Polymarket price points")
        except Exception as e:
            logger.warning(f"Could not load Polymarket prices: {e}")
            self.prices_poly = pd.DataFrame()

        try:
            self.prices_kalshi = pd.read_parquet(
                f"{self.storage.base_path}/prices/kalshi/"
            )
            logger.info(f"Loaded {len(self.prices_kalshi):,} Kalshi price points")
        except Exception as e:
            logger.warning(f"Could not load Kalshi prices: {e}")
            self.prices_kalshi = pd.DataFrame()

        try:
            self.markets_poly = pd.read_parquet(
                f"{self.storage.base_path}/markets/polymarket/markets.parquet"
            )
            logger.info(f"Loaded {len(self.markets_poly):,} Polymarket markets")
        except Exception as e:
            logger.warning(f"Could not load Polymarket markets: {e}")
            self.markets_poly = pd.DataFrame()

        try:
            self.markets_kalshi = pd.read_parquet(
                f"{self.storage.base_path}/markets/kalshi/markets.parquet"
            )
            logger.info(f"Loaded {len(self.markets_kalshi):,} Kalshi markets")
        except Exception as e:
            logger.warning(f"Could not load Kalshi markets: {e}")
            self.markets_kalshi = pd.DataFrame()

    def get_combined_prices(self, filter_quality: bool = True) -> pd.DataFrame:
        """
        Get combined price data from both platforms.

        Args:
            filter_quality: If True, filter out low-quality data points
                           (e.g., Kalshi placeholder 0.50 prices with no volume)
        """
        dfs = []
        if self.prices_poly is not None and len(self.prices_poly) > 0:
            df = self.prices_poly.copy()
            df['platform'] = 'Polymarket'
            dfs.append(df)
        if self.prices_kalshi is not None and len(self.prices_kalshi) > 0:
            df = self.prices_kalshi.copy()
            df['platform'] = 'Kalshi'

            if filter_quality:
                # Filter out Kalshi placeholder data:
                # - price_yes == 0.50 with volume == 0 and no OHLC data
                original_len = len(df)
                has_ohlc = df['open'].notna() | df['close'].notna()
                has_volume = (df['volume'].fillna(0) > 0)
                not_placeholder = (df['price_yes'] != 0.50) | has_ohlc | has_volume
                df = df[not_placeholder]
                filtered = original_len - len(df)
                if filtered > 0:
                    logger.info(f"Filtered {filtered:,} Kalshi placeholder price points")

            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute returns for each market.

        Args:
            prices: Price DataFrame with market_id and price_yes columns

        Returns:
            DataFrame with added 'return' column
        """
        df = prices.copy()
        df = df.sort_values(['market_id', 'timestamp'])
        df['return'] = df.groupby('market_id')['price_yes'].pct_change()
        df['log_return'] = np.log(df['price_yes'] / df.groupby('market_id')['price_yes'].shift(1))
        return df

    def compute_market_stats(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute summary statistics for each market.

        Args:
            prices: Price DataFrame

        Returns:
            DataFrame with market-level statistics
        """
        df = self.compute_returns(prices)

        stats_list = []
        for market_id, group in df.groupby('market_id'):
            if len(group) < 5:
                continue

            returns = group['return'].dropna()
            prices_series = group['price_yes'].dropna()

            stats_dict = {
                'market_id': market_id,
                'n_observations': len(group),
                'mean_price': prices_series.mean(),
                'std_price': prices_series.std(),
                'min_price': prices_series.min(),
                'max_price': prices_series.max(),
                'price_range': prices_series.max() - prices_series.min(),
                'start_price': prices_series.iloc[0] if len(prices_series) > 0 else np.nan,
                'end_price': prices_series.iloc[-1] if len(prices_series) > 0 else np.nan,
            }

            if len(returns) > 1:
                stats_dict.update({
                    'mean_return': returns.mean(),
                    'std_return': returns.std(),
                    'volatility': returns.std() * np.sqrt(24),  # Annualized (hourly data)
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'max_drawdown': self._compute_max_drawdown(prices_series),
                })
            else:
                stats_dict.update({
                    'mean_return': np.nan,
                    'std_return': np.nan,
                    'volatility': np.nan,
                    'skewness': np.nan,
                    'kurtosis': np.nan,
                    'max_drawdown': np.nan,
                })

            stats_list.append(stats_dict)

        return pd.DataFrame(stats_list)

    @staticmethod
    def _compute_max_drawdown(prices: pd.Series) -> float:
        """Compute maximum drawdown from price series."""
        if len(prices) < 2:
            return 0.0
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return drawdown.min()

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def plot_price_trajectories(
        self,
        n_markets: int = 10,
        platform: Optional[str] = None,
    ) -> None:
        """
        Plot price trajectories for top markets by data points.

        Args:
            n_markets: Number of markets to plot
            platform: Filter by platform ('polymarket', 'kalshi', or None for both)
        """
        logger.info(f"Plotting price trajectories for top {n_markets} markets...")

        prices = self.get_combined_prices()
        if len(prices) == 0:
            logger.warning("No price data available")
            return

        if platform:
            prices = prices[prices['platform'].str.lower() == platform.lower()]

        # Get top markets by observation count
        market_counts = prices.groupby('market_id').size().sort_values(ascending=False)
        top_markets = market_counts.head(n_markets).index.tolist()

        fig, axes = plt.subplots(
            (n_markets + 1) // 2, 2,
            figsize=(14, 3 * ((n_markets + 1) // 2)),
            squeeze=False,
        )
        axes = axes.flatten()

        for idx, market_id in enumerate(top_markets):
            ax = axes[idx]
            market_data = prices[prices['market_id'] == market_id].sort_values('timestamp')

            ax.plot(
                market_data['timestamp'],
                market_data['price_yes'],
                linewidth=1.5,
                alpha=0.8,
            )
            ax.fill_between(
                market_data['timestamp'],
                market_data['price_yes'],
                alpha=0.3,
            )

            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            ax.set_title(f'{market_id[:30]}... ({len(market_data):,} pts)', fontsize=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.tick_params(axis='x', rotation=45)

        # Hide unused subplots
        for idx in range(len(top_markets), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Price Trajectories - Top Markets by Data Points', fontsize=14, y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / 'price_trajectories.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")

    def plot_price_distribution(self) -> None:
        """Plot distribution of prices across all markets."""
        logger.info("Plotting price distribution...")

        prices = self.get_combined_prices()
        if len(prices) == 0:
            logger.warning("No price data available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Overall distribution
        ax1 = axes[0]
        for platform in prices['platform'].unique():
            platform_data = prices[prices['platform'] == platform]['price_yes']
            ax1.hist(
                platform_data,
                bins=50,
                alpha=0.6,
                label=f'{platform} (n={len(platform_data):,})',
                density=True,
            )
        ax1.set_xlabel('Price (Probability)')
        ax1.set_ylabel('Density')
        ax1.set_title('Price Distribution by Platform')
        ax1.legend()
        ax1.set_xlim(0, 1)

        # KDE plot
        ax2 = axes[1]
        for platform in prices['platform'].unique():
            platform_data = prices[prices['platform'] == platform]['price_yes']
            if len(platform_data) > 100:
                sns.kdeplot(data=platform_data, ax=ax2, label=platform, fill=True, alpha=0.3)
        ax2.set_xlabel('Price (Probability)')
        ax2.set_ylabel('Density')
        ax2.set_title('Price Distribution (KDE)')
        ax2.legend()
        ax2.set_xlim(0, 1)

        plt.tight_layout()
        output_path = self.output_dir / 'price_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")

    def plot_return_distribution(self) -> None:
        """Plot distribution of returns."""
        logger.info("Plotting return distribution...")

        prices = self.get_combined_prices()
        if len(prices) == 0:
            logger.warning("No price data available")
            return

        prices = self.compute_returns(prices)
        returns = prices['return'].dropna()
        returns = returns[(returns > -0.5) & (returns < 0.5)]  # Filter outliers

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Histogram
        ax1 = axes[0]
        ax1.hist(returns, bins=100, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Overlay normal distribution
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')

        ax1.set_xlabel('Return')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Return Distribution\nμ={mu:.4f}, σ={sigma:.4f}')
        ax1.legend()

        # Q-Q plot
        ax2 = axes[1]
        stats.probplot(returns.sample(min(10000, len(returns))), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot vs Normal')

        # By platform
        ax3 = axes[2]
        for platform in prices['platform'].unique():
            platform_returns = prices[prices['platform'] == platform]['return'].dropna()
            platform_returns = platform_returns[(platform_returns > -0.5) & (platform_returns < 0.5)]
            if len(platform_returns) > 100:
                sns.kdeplot(data=platform_returns, ax=ax3, label=platform, fill=True, alpha=0.3)

        ax3.set_xlabel('Return')
        ax3.set_ylabel('Density')
        ax3.set_title('Return Distribution by Platform')
        ax3.legend()

        plt.tight_layout()
        output_path = self.output_dir / 'return_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")

    def plot_volatility_analysis(self) -> None:
        """Plot volatility analysis across markets."""
        logger.info("Plotting volatility analysis...")

        prices = self.get_combined_prices()
        if len(prices) == 0:
            logger.warning("No price data available")
            return

        market_stats = self.compute_market_stats(prices)
        market_stats = market_stats.dropna(subset=['volatility'])

        if len(market_stats) == 0:
            logger.warning("Not enough data for volatility analysis")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Volatility distribution
        ax1 = axes[0, 0]
        ax1.hist(market_stats['volatility'], bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(market_stats['volatility'].median(), color='red', linestyle='--',
                    label=f'Median: {market_stats["volatility"].median():.3f}')
        ax1.set_xlabel('Volatility (Annualized)')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Market Volatility')
        ax1.legend()

        # Volatility vs number of observations
        ax2 = axes[0, 1]
        ax2.scatter(market_stats['n_observations'], market_stats['volatility'], alpha=0.5, s=20)
        ax2.set_xlabel('Number of Observations')
        ax2.set_ylabel('Volatility')
        ax2.set_title('Volatility vs Data Points')
        ax2.set_xscale('log')

        # Volatility vs mean price
        ax3 = axes[1, 0]
        ax3.scatter(market_stats['mean_price'], market_stats['volatility'], alpha=0.5, s=20)
        ax3.set_xlabel('Mean Price')
        ax3.set_ylabel('Volatility')
        ax3.set_title('Volatility vs Price Level')
        ax3.set_xlim(0, 1)

        # Max drawdown distribution
        ax4 = axes[1, 1]
        drawdowns = market_stats['max_drawdown'].dropna()
        ax4.hist(drawdowns, bins=50, edgecolor='black', alpha=0.7)
        ax4.axvline(drawdowns.median(), color='red', linestyle='--',
                    label=f'Median: {drawdowns.median():.3f}')
        ax4.set_xlabel('Max Drawdown')
        ax4.set_ylabel('Count')
        ax4.set_title('Distribution of Max Drawdowns')
        ax4.legend()

        plt.tight_layout()
        output_path = self.output_dir / 'volatility_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")

    def plot_platform_comparison(self) -> None:
        """Compare statistics between platforms."""
        logger.info("Plotting platform comparison...")

        prices = self.get_combined_prices()
        if len(prices) == 0:
            logger.warning("No price data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Data volume comparison
        ax1 = axes[0, 0]
        platform_counts = prices.groupby('platform').size()
        ax1.bar(platform_counts.index, platform_counts.values, color=['#2ecc71', '#3498db'])
        ax1.set_ylabel('Number of Price Points')
        ax1.set_title('Data Volume by Platform')
        for i, v in enumerate(platform_counts.values):
            ax1.text(i, v + 1000, f'{v:,}', ha='center', fontsize=10)

        # Markets count
        ax2 = axes[0, 1]
        market_counts = prices.groupby('platform')['market_id'].nunique()
        ax2.bar(market_counts.index, market_counts.values, color=['#2ecc71', '#3498db'])
        ax2.set_ylabel('Number of Markets')
        ax2.set_title('Markets by Platform')
        for i, v in enumerate(market_counts.values):
            ax2.text(i, v + 5, f'{v:,}', ha='center', fontsize=10)

        # Price distribution comparison (box plot)
        ax3 = axes[1, 0]
        prices_sample = prices.sample(min(50000, len(prices)))
        sns.boxplot(data=prices_sample, x='platform', y='price_yes', ax=ax3)
        ax3.set_ylabel('Price')
        ax3.set_title('Price Distribution by Platform')

        # Average observations per market
        ax4 = axes[1, 1]
        # Calculate average observations per market for each platform
        avg_obs_dict = {}
        for platform in prices['platform'].unique():
            platform_prices = prices[prices['platform'] == platform]
            market_sizes = platform_prices.groupby('market_id').size()
            avg_obs_dict[platform] = market_sizes.mean()
        avg_obs = pd.Series(avg_obs_dict)
        ax4.bar(avg_obs.index, avg_obs.values, color=['#2ecc71', '#3498db'][:len(avg_obs)])
        ax4.set_ylabel('Avg Observations per Market')
        ax4.set_title('Data Density by Platform')
        for i, v in enumerate(avg_obs.values):
            ax4.text(i, v + 10, f'{v:.0f}', ha='center', fontsize=10)

        plt.tight_layout()
        output_path = self.output_dir / 'platform_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")

    def plot_time_patterns(self) -> None:
        """Analyze temporal patterns in the data."""
        logger.info("Plotting time patterns...")

        prices = self.get_combined_prices()
        if len(prices) == 0:
            logger.warning("No price data available")
            return

        # Ensure timestamp is datetime
        prices['timestamp'] = pd.to_datetime(prices['timestamp'])
        prices['hour'] = prices['timestamp'].dt.hour
        prices['day_of_week'] = prices['timestamp'].dt.dayofweek
        prices['date'] = prices['timestamp'].dt.date

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        # Observations by hour
        ax1 = axes[0, 0]
        hourly = prices.groupby('hour').size()
        ax1.bar(hourly.index, hourly.values, color='steelblue', alpha=0.8)
        ax1.set_xlabel('Hour of Day (UTC)')
        ax1.set_ylabel('Number of Observations')
        ax1.set_title('Trading Activity by Hour')
        ax1.set_xticks(range(0, 24, 2))

        # Observations by day of week
        ax2 = axes[0, 1]
        daily = prices.groupby('day_of_week').size().reindex(range(7), fill_value=0)
        ax2.bar(range(7), daily.values, color='steelblue', alpha=0.8)
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(days)
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Number of Observations')
        ax2.set_title('Trading Activity by Day')

        # Timeline of data collection
        ax3 = axes[1, 0]
        date_counts = prices.groupby('date').size()
        dates = pd.to_datetime(list(date_counts.index))
        ax3.plot(dates, date_counts.values, linewidth=1)
        ax3.fill_between(dates, date_counts.values, alpha=0.3)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Observations')
        ax3.set_title('Data Collection Timeline')
        ax3.tick_params(axis='x', rotation=45)

        # Heatmap of hour vs day (normalize by row to show relative patterns)
        ax4 = axes[1, 1]
        heatmap_data = prices.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        # Ensure all days are present
        heatmap_data = heatmap_data.reindex(range(7), fill_value=0)
        sns.heatmap(
            heatmap_data,
            ax=ax4,
            cmap='YlOrRd',
            yticklabels=days,
            cbar_kws={'label': 'Observations'},
        )
        ax4.set_xlabel('Hour of Day (UTC)')
        ax4.set_ylabel('Day of Week')
        ax4.set_title('Activity Heatmap\n(Note: Hour 0 UTC = 7pm EST / 4pm PST)')

        plt.tight_layout()
        output_path = self.output_dir / 'time_patterns.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")

    def generate_summary_stats(self) -> pd.DataFrame:
        """Generate summary statistics table."""
        logger.info("Generating summary statistics...")

        prices = self.get_combined_prices()
        if len(prices) == 0:
            return pd.DataFrame()

        summary = []

        for platform in ['Polymarket', 'Kalshi', 'Combined']:
            if platform == 'Combined':
                df = prices
            else:
                df = prices[prices['platform'] == platform]

            if len(df) == 0:
                continue

            df_returns = self.compute_returns(df)

            summary.append({
                'Platform': platform,
                'Total Price Points': len(df),
                'Unique Markets': df['market_id'].nunique(),
                'Avg Points per Market': len(df) / df['market_id'].nunique(),
                'Mean Price': df['price_yes'].mean(),
                'Std Price': df['price_yes'].std(),
                'Mean Return': df_returns['return'].mean(),
                'Std Return': df_returns['return'].std(),
                'Return Skewness': df_returns['return'].skew(),
                'Return Kurtosis': df_returns['return'].kurtosis(),
            })

        return pd.DataFrame(summary)

    def run_all_analyses(self) -> None:
        """Run all EDA analyses and generate all plots."""
        logger.info("Running all EDA analyses...")

        # Load data
        self.load_data()

        # Check if we have data
        prices = self.get_combined_prices()
        if len(prices) == 0:
            logger.error("No data available for analysis")
            return

        # Generate all plots
        self.plot_price_trajectories(n_markets=10)
        self.plot_price_distribution()
        self.plot_return_distribution()
        self.plot_volatility_analysis()
        self.plot_platform_comparison()
        self.plot_time_patterns()

        # Generate summary stats
        summary = self.generate_summary_stats()
        if len(summary) > 0:
            summary_path = self.output_dir / 'summary_statistics.csv'
            summary.to_csv(summary_path, index=False)
            logger.info(f"Saved: {summary_path}")

            print("\n" + "=" * 60)
            print("SUMMARY STATISTICS")
            print("=" * 60)
            print(summary.to_string(index=False))
            print("=" * 60)

        logger.info(f"All plots saved to: {self.output_dir}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Exploratory Data Analysis for Prediction Markets",
    )

    parser.add_argument(
        "--storage-path",
        default="data/store",
        help="Path to data storage",
    )

    parser.add_argument(
        "--output-dir",
        default="output/plots",
        help="Directory for output plots",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    eda = PredictionMarketEDA(
        storage_path=args.storage_path,
        output_dir=args.output_dir,
    )

    eda.run_all_analyses()


if __name__ == "__main__":
    main()
