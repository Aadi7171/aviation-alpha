"""
BTS T-100 Domestic Segment Data Loader
========================================
Downloads and parses Bureau of Transportation Statistics T-100 data.
Uses synthetic/simulated data when offline (no BTS download required).

BTS Real data URL pattern (public):
  https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FLL

We simulate realistic BTS-style data so the pipeline runs fully offline.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import CONFIG, ROOT, ensure_dirs, get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CARRIERS = CONFIG["airlines"]["bts_codes"]
AIRPORTS = [
    "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "LAS", "MIA", "IAH",
    "SEA", "CLT", "PHX", "MCO", "EWR", "MSP", "BOS", "DTW", "PHL", "LGA",
    "FLL", "BWI", "SLC", "IAD", "MDW", "HNL", "SAN", "TPA", "PDX", "STL",
    "BNA", "HOU", "AUS", "MSY", "MKE", "RDU", "PIT", "OAK", "SMF", "SJC",
]

np.random.seed(42)


def _simulate_bts(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Generate synthetic BTS T-100 style monthly segment data.

    Columns align with real BTS T-100 schema:
      YEAR, MONTH, CARRIER, ORIGIN, DEST,
      DEPARTURES_PERFORMED, SEATS, PASSENGERS,
      AIR_TIME, DISTANCE,
      CANCELLED, DIVERTED, DEP_DELAY_MINUTES
    """
    logger.info("Generating synthetic BTS data (%d–%d)…", start_year, end_year)
    records = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            for carrier in tqdm(CARRIERS, desc=f"{year}-{month:02d}", leave=False):
                # Each carrier operates a subset of routes
                rng = np.random.default_rng(
                    int(hashlib.md5(f"{carrier}{year}{month}".encode()).hexdigest(), 16) % (2**32)
                )
                n_routes = rng.integers(15, 35)
                origins  = rng.choice(AIRPORTS, size=n_routes, replace=False)
                dests    = rng.choice(AIRPORTS, size=n_routes, replace=False)

                for orig, dest in zip(origins, dests):
                    if orig == dest:
                        continue
                    deps = int(rng.integers(20, 120))
                    seats = deps * int(rng.integers(100, 180))
                    load  = rng.uniform(0.70, 0.95)
                    pax   = int(seats * load)

                    # Inject seasonal disruption
                    base_cancel_rate = 0.015
                    if month in (1, 2, 12):          # winter spike
                        base_cancel_rate *= rng.uniform(2.0, 4.5)
                    if month in (6, 7, 8):           # summer storms
                        base_cancel_rate *= rng.uniform(1.2, 2.0)

                    cancelled = int(deps * base_cancel_rate)
                    delay_mins = float(rng.exponential(18.0) + base_cancel_rate * 60)

                    records.append({
                        "YEAR":                  year,
                        "MONTH":                 month,
                        "CARRIER":               carrier,
                        "ORIGIN":                orig,
                        "DEST":                  dest,
                        "DEPARTURES_PERFORMED":  deps,
                        "SEATS":                 seats,
                        "PASSENGERS":            pax,
                        "DISTANCE":              int(rng.integers(200, 2800)),
                        "CANCELLED":             cancelled,
                        "DIVERTED":              max(0, int(deps * 0.003)),
                        "DEP_DELAY_MINUTES":     round(delay_mins, 2),
                    })

    df = pd.DataFrame(records)
    df["DATE"] = pd.to_datetime(
        df[["YEAR", "MONTH"]].assign(DAY=1)
    )
    df["cancel_rate"] = df["CANCELLED"] / df["DEPARTURES_PERFORMED"].clip(lower=1)
    df["load_factor"]  = df["PASSENGERS"] / df["SEATS"].clip(lower=1)
    return df


def load_bts(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load BTS data. Uses cached parquet if available, else simulates.

    Args:
        force_refresh: If True, regenerate even if cache exists.

    Returns:
        DataFrame with monthly segment-level BTS data.
    """
    ensure_dirs(CONFIG)
    cache_path = ROOT / CONFIG["data"]["processed_dir"] / "bts_segments.parquet"

    if cache_path.exists() and not force_refresh:
        logger.info("Loading BTS data from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    df = _simulate_bts(
        start_year=CONFIG["data"]["start_year"],
        end_year=CONFIG["data"]["end_year"],
    )

    df.to_parquet(cache_path, index=False)
    logger.info("BTS data saved → %s  (%d rows)", cache_path, len(df))
    return df


def monthly_carrier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate segment-level BTS data to monthly carrier-level summary.

    Returns DataFrame indexed by (DATE, CARRIER) with:
      total_departures, cancel_rate, avg_delay, load_factor, n_routes
    """
    grp = df.groupby(["DATE", "CARRIER"])
    summary = grp.agg(
        total_departures   = ("DEPARTURES_PERFORMED", "sum"),
        total_cancelled    = ("CANCELLED", "sum"),
        total_seats        = ("SEATS", "sum"),
        total_passengers   = ("PASSENGERS", "sum"),
        avg_delay          = ("DEP_DELAY_MINUTES", "mean"),
        n_routes           = ("ORIGIN", "count"),
    ).reset_index()

    summary["cancel_rate"]  = summary["total_cancelled"] / summary["total_departures"].clip(lower=1)
    summary["load_factor"]  = summary["total_passengers"] / summary["total_seats"].clip(lower=1)
    summary["DATE"] = pd.to_datetime(summary["DATE"])
    return summary.sort_values(["DATE", "CARRIER"]).reset_index(drop=True)


if __name__ == "__main__":
    df = load_bts()
    summary = monthly_carrier_summary(df)
    print(summary.head(20).to_string())
    print(f"\nShape: {summary.shape}")
