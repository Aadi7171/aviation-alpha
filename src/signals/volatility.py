"""
Airline Stock Volatility Signal
==================================
Downloads airline stock prices via yfinance and computes:
  - 21-day realized volatility (rolling std of log returns)
  - VIX correlation
  - Airline sector volatility index (XAL)

Falls back to synthetic data if yfinance is unavailable or rate-limited.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.utils import CONFIG, get_logger

logger = get_logger(__name__)

TICKERS    = list(CONFIG["airlines"]["tickers"].keys()) + ["^VIX", "XAL"]
START_DATE = f"{CONFIG['data']['start_year']}-01-01"
END_DATE   = f"{CONFIG['data']['end_year']}-12-31"
VOL_WINDOW = CONFIG["features"]["volatility"]["realized_vol_window"]


def _download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame | None:
    """Try to download price data via yfinance."""
    try:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            return None
        # Extract Close prices
        if isinstance(df.columns, pd.MultiIndex):
            prices = df["Close"]
        else:
            prices = df[["Close"]]
        return prices.ffill().dropna(how="all")
    except Exception as exc:
        logger.warning("yfinance download failed: %s", exc)
        return None


def _synthetic_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Generate synthetic GBM price paths when yfinance is unavailable."""
    logger.info("Generating synthetic stock prices…")
    dates = pd.bdate_range(start=start, end=end)
    rng   = np.random.default_rng(777)
    data  = {}
    for tick in tickers:
        mu    = rng.uniform(-0.0002, 0.0008)
        sigma = rng.uniform(0.012, 0.035)
        S0    = rng.uniform(10, 200)
        log_r = rng.normal(mu, sigma, size=len(dates))
        prices = S0 * np.exp(np.cumsum(log_r))
        data[tick] = prices
    return pd.DataFrame(data, index=dates)


def get_airline_volatility(force_synthetic: bool = False) -> pd.DataFrame:
    """
    Get realised volatility for airline tickers + VIX.

    Args:
        force_synthetic: Skip yfinance and use synthetic data.

    Returns:
        DataFrame with [Date, <ticker>_vol, …, market_vol].
    """
    prices = None if force_synthetic else _download_prices(TICKERS, START_DATE, END_DATE)
    if prices is None:
        prices = _synthetic_prices(TICKERS, START_DATE, END_DATE)

    # Log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Rolling realised volatility
    vol = log_returns.rolling(VOL_WINDOW).std() * np.sqrt(252)  # annualised
    vol.columns = [f"{c}_vol" for c in vol.columns]
    vol = vol.dropna(how="all").reset_index()
    vol.rename(columns={"index": "Date"}, inplace=True)
    if "Date" not in vol.columns and vol.index.name == "Date":
        vol = vol.reset_index()

    # Market vol proxy: mean of airline vols (exclude VIX col)
    airline_vol_cols = [c for c in vol.columns if c.endswith("_vol") and "VIX" not in c and "XAL" not in c]
    vol["market_vol"] = vol[airline_vol_cols].mean(axis=1)

    return vol


def compute_vol_disruption_correlation(
    vol_df: pd.DataFrame,
    disruption_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute rolling correlation between market disruption index and market vol.

    Args:
        vol_df:         Output of get_airline_volatility().
        disruption_df:  Output of disruption.aggregate_market_disruption().

    Returns:
        DataFrame with [Date, rolling_correlation].
    """
    vol_monthly   = vol_df.set_index("Date")["market_vol"].resample("ME").mean()
    dis_monthly   = disruption_df.set_index("DATE")["market_disruption_index"].resample("ME").mean()

    combined = pd.DataFrame({
        "vol":        vol_monthly,
        "disruption": dis_monthly,
    }).dropna()

    combined["rolling_corr"] = (
        combined["vol"].rolling(6).corr(combined["disruption"])
    )
    return combined.reset_index()


if __name__ == "__main__":
    vol = get_airline_volatility()
    print("=== Airline Volatility Sample ===")
    print(vol.tail(10).to_string())
    print(f"\nColumns: {list(vol.columns)}")
