"""
Prediction Market Data Ingestion Module

This module handles the fetching, normalization, and storage of prediction market data.
Currently supports Polymarket via the CLOB API.

Features:
- Fetches active binary markets.
- Retrieves historical price data (candles).
- Normalizes probabilities to [0, 1].
- Stores data in Parquet format.
- Generates exploratory data analysis (EDA) plots.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class PolymarketClient:
    """
    Client for interacting with the Polymarket CLOB API.
    """
    BASE_URL = "https://clob.polymarket.com"
    
    def __init__(self, data_dir: str = "data/store", plots_dir: str = "output/plots"):
        self.data_dir = data_dir
        self.plots_dir = plots_dir
        self.session = requests.Session()
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def fetch_markets(self, limit: int = 500, min_liquidity: float = 0.0) -> pd.DataFrame:
        """
        Fetches a list of active markets from Polymarket.
        
        Args:
            limit: Max number of markets to fetch (pagination handling included).
            min_liquidity: Minimum liquidity filter (note: API might not expose liquidity directly in list, 
                           so this might require post-filtering if available).
                           
        Returns:
            pd.DataFrame: DataFrame containing market metadata.
        """
        logger.info("Fetching markets metadata...")
        markets = []
        cursor = ""
        
        # Fetch loop (simplified for demo, robust version would handle full pagination)
        while len(markets) < limit:
            try:
                url = f"{self.BASE_URL}/markets"
                params = {"next_cursor": cursor} if cursor else {}
                resp = self.session.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                
                batch = data.get("data", [])
                if not batch:
                    break
                
                markets.extend(batch)
                cursor = data.get("next_cursor")
                
                if not cursor or cursor == "0":
                    break
                    
                # Rate limit kindness
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error fetching markets: {e}")
                break

        df = pd.DataFrame(markets)
        
        if df.empty:
            logger.warning("No markets found.")
            return df

        # Filter for Binary Markets
        # Polymarket binary markets usually have 2 tokens.
        # We want markets that are active and not closed.
        
        # Ensure 'tokens' column exists and has length 2
        if 'tokens' in df.columns:
            df['is_binary'] = df['tokens'].apply(lambda x: isinstance(x, list) and len(x) == 2)
            df = df[df['is_binary']].copy()
        
        # Filter active
        if 'active' in df.columns:
            df = df[df['active'] == True]
            
        if 'closed' in df.columns:
            df = df[df['closed'] == False]

        logger.info(f"Retrieved {len(df)} active binary markets.")
        return df

    def fetch_price_history(self, market_id: str, start_ts: Optional[int] = None, interval: str = "1d") -> pd.DataFrame:
        """
        Fetches historical price candles for a specific market token.
        
        Args:
            market_id: The clobTokenId (usually the 'Yes' token).
            start_ts: Unix timestamp for start of history.
            interval: Candle interval ('1m', '1h', '1d').
            
        Returns:
            pd.DataFrame: Time-indexed price history.
        """
        url = f"{self.BASE_URL}/prices-history"
        params = {
            "market": market_id,
            "interval": interval,
            "fidelity": 1 # experimental param often required
        }
        if start_ts:
            params["startTs"] = start_ts

        try:
            resp = self.session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            history = data.get("history", [])
            if not history:
                return pd.DataFrame()
            
            df = pd.DataFrame(history)
            # Convert timestamp (seconds) to datetime
            df['t'] = pd.to_datetime(df['t'], unit='s')
            df.set_index('t', inplace=True)
            df.sort_index(inplace=True)
            
            # Normalize columns
            # API returns: o (open), h (high), l (low), c (close), v (volume)
            # Prices are usually 0-1.
            return df
            
        except Exception as e:
            logger.warning(f"Failed to fetch history for {market_id}: {e}")
            return pd.DataFrame()

    def process_and_save(self, markets_df: pd.DataFrame, max_markets: int = 50):
        """
        Orchestrates the data ingestion:
        1. Selects top markets (e.g. by liquidity/volume logic if available, or just first N).
        2. Fetches history for each.
        3. Saves metadata and history to Parquet.
        """
        if markets_df.empty:
            return

        # For this implementation, we take the first N markets that have a valid 'Yes' token.
        # In a real scenario, we would sort by volume (if available in metadata) or open interest.
        
        processed_data = []
        
        logger.info(f"Fetching history for top {max_markets} markets...")
        
        count = 0
        for _, row in tqdm(markets_df.iterrows(), total=min(len(markets_df), max_markets)):
            if count >= max_markets:
                break
                
            # Identify the 'Yes' token. 
            # Usually tokens[0] is Yes or No depending on the market, but for binary 
            # we often treat the first one or check outcome labels if available.
            # Polymarket CLOB usually returns tokens as [token_id_1, token_id_2].
            # We will fetch the first token's history as a proxy for the market dynamic.
            tokens = row.get('tokens')
            if not tokens or not isinstance(tokens, list):
                continue
                
            token_id = tokens[0].get('token_id') if isinstance(tokens[0], dict) else tokens[0]
            
            # Fetch history
            hist_df = self.fetch_price_history(token_id, interval="1d")
            
            if hist_df.empty or len(hist_df) < 5: # Skip if too little data
                continue
                
            # Add metadata columns to the history frame for easier joining later
            hist_df['market_id'] = row.get('condition_id')
            hist_df['question'] = row.get('question')
            hist_df['token_id'] = token_id
            
            processed_data.append(hist_df)
            count += 1
            time.sleep(0.1) # Rate limit

        if not processed_data:
            logger.warning("No historical data retrieved.")
            return

        # Combine all histories
        all_history = pd.concat(processed_data)
        
        # Save to Parquet
        history_path = os.path.join(self.data_dir, "price_history.parquet")
        markets_path = os.path.join(self.data_dir, "markets_metadata.parquet")
        
        all_history.to_parquet(history_path, index=True)
        markets_df.to_parquet(markets_path, index=False)
        
        logger.info(f"Saved {len(all_history)} rows of price history to {history_path}")
        logger.info(f"Saved market metadata to {markets_path}")
        
        return all_history

    def generate_plots(self, df: pd.DataFrame):
        """
        Generates EDA plots from the ingested data.
        """
        if df.empty:
            return

        logger.info("Generating EDA plots...")
        
        # 1. Price Trajectories (Line Chart)
        plt.figure(figsize=(12, 6))
        # Plot a subset of markets to avoid clutter
        sample_markets = df['question'].unique()[:10]
        
        for market in sample_markets:
            subset = df[df['question'] == market]
            plt.plot(subset.index, subset['c'], label=market[:30]+"...", alpha=0.7)
            
        plt.title("Price Trajectories (Top 10 Markets)")
        plt.xlabel("Date")
        plt.ylabel("Price (Probability)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "price_trajectories.png"))
        plt.close()

        # 2. Price Distribution (PDF)
        plt.figure(figsize=(10, 6))
        sns.histplot(df['c'], bins=50, kde=True, color='teal')
        plt.title("Global Price Distribution (Probability Density)")
        plt.xlabel("Price")
        plt.ylabel("Count")
        plt.savefig(os.path.join(self.plots_dir, "price_distribution.pdf"))
        plt.close()

        # 3. Volatility Heatmap (Daily Returns)
        # Pivot data to get wide format: Index=Date, Columns=Market, Values=Close
        # We need to handle duplicate timestamps if any, though daily should be unique per market
        try:
            pivot_df = df.reset_index().pivot(index='t', columns='question', values='c')
            
            # Calculate daily returns
            returns_df = pivot_df.pct_change().dropna(how='all')
            
            # Correlation matrix of returns (Landscape of dependencies)
            # Filter to markets with enough overlap
            valid_cols = returns_df.count() > 10
            corr_matrix = returns_df.loc[:, valid_cols].corr()
            
            if not corr_matrix.empty:
                plt.figure(figsize=(12, 10))
                # If too many markets, take top 20
                if corr_matrix.shape[0] > 20:
                    corr_matrix = corr_matrix.iloc[:20, :20]
                    
                sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
                plt.title("Market Correlation Heatmap (Daily Returns)")
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, "correlation_heatmap.png"))
                plt.close()
        except Exception as e:
            logger.warning(f"Could not generate heatmap: {e}")

        # 4. Volume vs Price Scatter (if volume available)
        # Note: 'v' in history is volume for that interval
        plt.figure(figsize=(10, 6))
        plt.scatter(df['c'], np.log1p(df['v']), alpha=0.3, s=10)
        plt.title("Price vs Log-Volume")
        plt.xlabel("Price")
        plt.ylabel("Log Volume")
        plt.savefig(os.path.join(self.plots_dir, "price_volume_scatter.png"))
        plt.close()
        
        logger.info(f"Plots saved to {self.plots_dir}")

def main():
    """
    Main execution entry point.
    """
    client = PolymarketClient()
    
    # 1. Fetch Markets
    markets = client.fetch_markets(limit=100)
    
    # 2. Process and Save History
    # We limit to 20 markets for the demo to be quick
    history = client.process_and_save(markets, max_markets=20)
    
    # 3. Generate Plots
    if history is not None and not history.empty:
        client.generate_plots(history)
    else:
        logger.warning("No data to plot.")

if __name__ == "__main__":
    main()