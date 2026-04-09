"""
Route Overlap Feature Engineering
====================================
Computes pairwise Jaccard similarity of airline route sets over time.

Route overlap between two airlines is a proxy for:
  - Competitive pressure (high overlap → price wars or acquisition interest)
  - Market redundancy (potential consolidation target)

Jaccard(A, B) = |routes_A ∩ routes_B| / |routes_A ∪ routes_B|
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import CONFIG, ROOT, get_logger

logger = get_logger(__name__)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets of routes."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def build_overlap_matrix(
    carrier_routes: dict[str, set],
    carriers_of_interest: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a square pairwise Jaccard overlap matrix for all carriers.

    Args:
        carrier_routes:        Dict mapping carrier code → set of (src, dst) pairs.
        carriers_of_interest:  Subset of carriers to include. None = all.

    Returns:
        DataFrame (carriers × carriers) of Jaccard similarity scores.
    """
    if carriers_of_interest:
        carriers = [c for c in carriers_of_interest if c in carrier_routes]
    else:
        carriers = list(carrier_routes.keys())

    n = len(carriers)
    matrix = np.zeros((n, n))

    for i, ca in enumerate(carriers):
        for j, cb in enumerate(carriers):
            if i == j:
                matrix[i, j] = 1.0
            elif j > i:
                sim = jaccard_similarity(carrier_routes[ca], carrier_routes[cb])
                matrix[i, j] = sim
                matrix[j, i] = sim

    return pd.DataFrame(matrix, index=carriers, columns=carriers)


def temporal_overlap(
    bts_df: pd.DataFrame,
    carriers: list[str],
    window_quarters: int | None = None,
) -> pd.DataFrame:
    """
    Compute rolling Jaccard overlap per (quarter, carrier_pair).

    Args:
        bts_df:           BTS segment data with [DATE, CARRIER, ORIGIN, DEST].
        carriers:         Carriers to include.
        window_quarters:  Rolling window size in quarters.

    Returns:
        DataFrame with [period, carrier_a, carrier_b, jaccard, overlap_count].
    """
    wq = window_quarters or CONFIG["features"]["route_overlap"]["time_window_quarters"]

    bts_df = bts_df.copy()
    bts_df["PERIOD"] = pd.PeriodIndex(bts_df["DATE"], freq="Q")
    periods = sorted(bts_df["PERIOD"].unique())

    records = []
    for i, period in enumerate(periods):
        # Include last `wq` quarters in the rolling window
        window_periods = periods[max(0, i - wq + 1): i + 1]
        window_df = bts_df[bts_df["PERIOD"].isin(window_periods)]

        # Build route sets per carrier in this window
        route_sets: dict[str, set] = {}
        for carrier in carriers:
            sub = window_df[window_df["CARRIER"] == carrier]
            route_sets[carrier] = set(zip(sub["ORIGIN"], sub["DEST"]))

        for ca, cb in itertools.combinations(carriers, 2):
            j = jaccard_similarity(route_sets.get(ca, set()), route_sets.get(cb, set()))
            overlap_count = len(route_sets.get(ca, set()) & route_sets.get(cb, set()))
            records.append({
                "period":         str(period),
                "carrier_a":      ca,
                "carrier_b":      cb,
                "jaccard":        round(j, 4),
                "overlap_routes": overlap_count,
                "pair":           f"{ca}/{cb}",
            })

    return pd.DataFrame(records)


def top_overlapping_pairs(
    overlap_df: pd.DataFrame,
    n: int = 10,
    latest_only: bool = True,
) -> pd.DataFrame:
    """
    Return top-N most overlapping carrier pairs.

    Args:
        overlap_df:   Output of temporal_overlap().
        n:            Number of pairs to return.
        latest_only:  If True, only use the most recent period.

    Returns:
        Sorted DataFrame of top overlapping pairs.
    """
    if latest_only:
        latest = overlap_df["period"].max()
        df = overlap_df[overlap_df["period"] == latest]
    else:
        df = overlap_df.groupby("pair")["jaccard"].mean().reset_index()
        df.columns = ["pair", "jaccard"]

    return df.sort_values("jaccard", ascending=False).head(n).reset_index(drop=True)


if __name__ == "__main__":
    from src.ingestion.bts_loader import load_bts
    from src.ingestion.openflights import load_routes, carrier_route_sets

    routes = load_routes()
    crs = carrier_route_sets(routes)
    carriers = CONFIG["airlines"]["bts_codes"]

    # Static overlap matrix
    matrix = build_overlap_matrix(crs, carriers_of_interest=carriers)
    print("=== Static Jaccard Overlap Matrix ===")
    print(matrix.round(3).to_string())

    # Temporal overlap
    bts = load_bts()
    overlap = temporal_overlap(bts, carriers)
    print("\n=== Top 10 Overlapping Pairs (latest quarter) ===")
    print(top_overlapping_pairs(overlap).to_string())
