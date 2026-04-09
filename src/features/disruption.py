"""
Aviation Disruption Spike Detection
=======================================
Detects abnormal surges in cancellation rates and departure delays
using rolling z-score analysis.

Logic:
  - Compute 30-day rolling mean and std of each carrier's cancel_rate
  - z = (x - μ) / σ  →  |z| > 2.5 means disruption spike
  - Assign a composite disruption_score per carrier per day
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import CONFIG, get_logger

logger = get_logger(__name__)


def compute_disruption_timeseries(
    summary_df: pd.DataFrame,
    window: int | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Compute disruption z-score timeseries per carrier.

    Args:
        summary_df:  Output of bts_loader.monthly_carrier_summary().
                     Must have [DATE, CARRIER, cancel_rate, avg_delay].
        window:      Rolling window in months (default from config).
        threshold:   Z-score spike threshold (default from config).

    Returns:
        DataFrame with [DATE, CARRIER, cancel_rate, avg_delay,
                        cancel_zscore, delay_zscore,
                        disruption_score, spike].
    """
    win = window or CONFIG["features"]["disruption"]["zscore_window"]
    thr = threshold or CONFIG["features"]["disruption"]["spike_threshold"]

    dfs = []
    for carrier, grp in summary_df.groupby("CARRIER"):
        grp = grp.sort_values("DATE").copy().reset_index(drop=True)

        # Rolling z-score: cancellation rate
        roll_cancel = grp["cancel_rate"].rolling(win, min_periods=2)
        grp["cancel_zscore"] = (
            (grp["cancel_rate"] - roll_cancel.mean()) / roll_cancel.std().replace(0, np.nan)
        ).fillna(0)

        # Rolling z-score: average delay
        roll_delay = grp["avg_delay"].rolling(win, min_periods=2)
        grp["delay_zscore"] = (
            (grp["avg_delay"] - roll_delay.mean()) / roll_delay.std().replace(0, np.nan)
        ).fillna(0)

        # Composite disruption score (weighted average of absolute z-scores)
        grp["disruption_score"] = (
            0.6 * grp["cancel_zscore"].abs() + 0.4 * grp["delay_zscore"].abs()
        )

        # Binary spike flag
        grp["spike"] = (
            (grp["cancel_zscore"].abs() > thr) | (grp["delay_zscore"].abs() > thr)
        ).astype(int)

        dfs.append(grp)

    result = pd.concat(dfs, ignore_index=True)
    return result.sort_values(["DATE", "CARRIER"]).reset_index(drop=True)


def spike_event_calendar(disruption_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a table of all spike events with severity ranking.

    Args:
        disruption_df: Output of compute_disruption_timeseries().

    Returns:
        DataFrame of spike events sorted by severity.
    """
    spikes = disruption_df[disruption_df["spike"] == 1].copy()
    spikes["severity"] = pd.cut(
        spikes["disruption_score"],
        bins=[0, 2.5, 4.0, 6.0, np.inf],
        labels=["Moderate", "High", "Severe", "Extreme"],
    )
    return spikes[
        ["DATE", "CARRIER", "cancel_rate", "avg_delay", "disruption_score", "severity"]
    ].sort_values("disruption_score", ascending=False).reset_index(drop=True)


def aggregate_market_disruption(disruption_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-carrier disruption to a single market-wide disruption index.

    Useful for correlating with VIX or market volatility.

    Returns:
        DataFrame with [DATE, market_disruption_index, n_carriers_spiking].
    """
    grp = disruption_df.groupby("DATE")
    market = grp.agg(
        market_disruption_index = ("disruption_score", "mean"),
        n_carriers_spiking      = ("spike", "sum"),
        max_disruption          = ("disruption_score", "max"),
    ).reset_index()
    return market.sort_values("DATE").reset_index(drop=True)


if __name__ == "__main__":
    from src.ingestion.bts_loader import load_bts, monthly_carrier_summary

    bts = load_bts()
    summary = monthly_carrier_summary(bts)
    disruption = compute_disruption_timeseries(summary)

    print("=== Disruption Timeseries Sample ===")
    print(disruption[disruption["spike"] == 1].head(15).to_string())

    print("\n=== Spike Event Calendar ===")
    print(spike_event_calendar(disruption).head(10).to_string())

    print("\n=== Market Disruption Index ===")
    market = aggregate_market_disruption(disruption)
    print(market.tail(10).to_string())
