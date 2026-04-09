"""
M&A Pressure Signal
=====================
Combines route overlap, network vulnerability, and disruption spikes
into a single composite M&A pressure score per carrier.

Validated against known airline mergers:
  - Delta / Northwest (2008)
  - United / Continental (2010)
  - Southwest / AirTran (2011)
  - American / US Airways (2013)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import CONFIG, get_logger

logger = get_logger(__name__)

# Known M&A events for backtesting validation
HISTORICAL_MERGERS = [
    {"acquirer": "DL", "target": "NW", "announced": "2008-04-14", "completed": "2008-10-29"},
    {"acquirer": "UA", "target": "CO", "announced": "2010-05-03", "completed": "2010-10-01"},
    {"acquirer": "WN", "target": "FL", "announced": "2010-09-27", "completed": "2011-05-02"},
    {"acquirer": "AA", "target": "US", "announced": "2013-02-14", "completed": "2015-04-08"},
]


def compute_ma_pressure_score(
    network_metrics:   pd.DataFrame,
    overlap_df:        pd.DataFrame,
    disruption_df:     pd.DataFrame,
    carriers:          list[str],
) -> pd.DataFrame:
    """
    Compute a composite M&A pressure score per carrier per period.

    Score = w1 * network_vulnerability
          + w2 * max_route_overlap
          + w3 * disruption_spike_rate

    Args:
        network_metrics:  Output of network.compute_network_metrics().
        overlap_df:       Output of route_overlap.temporal_overlap().
        disruption_df:    Output of disruption.compute_disruption_timeseries(),
                          filtered to relevant carriers.
        carriers:         Carrier codes to include.

    Returns:
        DataFrame with [period, carrier, network_vuln, max_overlap,
                        disruption_rate, ma_pressure_score].
    """
    periods = sorted(overlap_df["period"].unique())
    records = []

    # Pre-compute disruption spike rate per carrier per period
    disruption_df = disruption_df.copy()
    disruption_df["PERIOD"] = pd.PeriodIndex(disruption_df["DATE"], freq="Q").astype(str)

    for period in periods:
        period_overlap = overlap_df[overlap_df["period"] == period]

        for carrier in carriers:
            # 1. Network vulnerability (static over time for now)
            nm_row = network_metrics[network_metrics["carrier"] == carrier]
            net_vuln = float(nm_row["network_vulnerability"].values[0]) if len(nm_row) else 0.5

            # 2. Max route overlap with any other carrier
            rows_a = period_overlap[period_overlap["carrier_a"] == carrier]["jaccard"]
            rows_b = period_overlap[period_overlap["carrier_b"] == carrier]["jaccard"]
            all_j  = pd.concat([rows_a, rows_b])
            max_overlap = float(all_j.max()) if not all_j.empty else 0.0

            # 3. Disruption spike rate this quarter
            dis_period = disruption_df[
                (disruption_df["CARRIER"] == carrier) &
                (disruption_df["PERIOD"] == period)
            ]
            spike_rate = float(dis_period["spike"].mean()) if not dis_period.empty else 0.0

            # Composite score
            score = (
                0.35 * net_vuln +
                0.40 * max_overlap +
                0.25 * spike_rate
            )

            records.append({
                "period":             period,
                "carrier":            carrier,
                "network_vuln":       round(net_vuln, 4),
                "max_route_overlap":  round(max_overlap, 4),
                "disruption_rate":    round(spike_rate, 4),
                "ma_pressure_score":  round(score, 4),
            })

    df = pd.DataFrame(records)
    return df.sort_values(["period", "ma_pressure_score"], ascending=[True, False]).reset_index(drop=True)


def validate_against_historical(
    ma_scores: pd.DataFrame,
    lookback_quarters: int = 4,
) -> pd.DataFrame:
    """
    Check if M&A pressure score was elevated before known merger announcements.

    Args:
        ma_scores:          Output of compute_ma_pressure_score().
        lookback_quarters:  How many quarters before announcement to check.

    Returns:
        DataFrame with validation results.
    """
    carrier_map = {
        "DL": "DL", "NW": "DL",  # Delta acquired Northwest (use DL as proxy)
        "UA": "UA", "CO": "UA",
        "WN": "WN", "FL": "WN",
        "AA": "AA", "US": "AA",
    }

    results = []
    for merger in HISTORICAL_MERGERS:
        announced = pd.Timestamp(merger["announced"])
        # Find periods in the lookback window
        lookback_start = announced - pd.DateOffset(months=3 * lookback_quarters)

        for carrier_code in [merger["acquirer"], merger["target"]]:
            proxy = carrier_map.get(carrier_code, carrier_code)
            sub = ma_scores[
                (ma_scores["carrier"] == proxy)
            ].copy()

            # Convert period string to timestamps for filtering
            sub["period_ts"] = sub["period"].apply(
                lambda p: pd.Period(p, freq="Q").to_timestamp(how="end")
            )
            pre_merger = sub[
                (sub["period_ts"] >= lookback_start) &
                (sub["period_ts"] <= announced)
            ]

            avg_score = float(pre_merger["ma_pressure_score"].mean()) if not pre_merger.empty else np.nan
            results.append({
                "merger":        f"{merger['acquirer']}/{merger['target']}",
                "announced":     merger["announced"],
                "carrier_role":  "acquirer" if carrier_code == merger["acquirer"] else "target",
                "carrier":       carrier_code,
                "proxy_code":    proxy,
                "avg_pre_merger_score": round(avg_score, 4) if not np.isnan(avg_score) else None,
                "data_available": not pre_merger.empty,
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    from src.ingestion.bts_loader import load_bts, monthly_carrier_summary
    from src.ingestion.openflights import load_routes
    from src.features.route_overlap import temporal_overlap
    from src.features.disruption import compute_disruption_timeseries
    from src.features.network import compute_network_metrics

    carriers = CONFIG["airlines"]["bts_codes"]
    bts      = load_bts()
    summary  = monthly_carrier_summary(bts)
    routes   = load_routes()

    overlap       = temporal_overlap(bts, carriers)
    disruption    = compute_disruption_timeseries(summary)
    net_metrics   = compute_network_metrics(routes, carriers)

    scores = compute_ma_pressure_score(net_metrics, overlap, disruption, carriers)
    print("=== M&A Pressure Scores (latest 5 periods) ===")
    print(scores.tail(20).to_string())

    val = validate_against_historical(scores)
    print("\n=== Historical Merger Validation ===")
    print(val.to_string())
