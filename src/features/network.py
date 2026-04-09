"""
Airline Route Network Graph Analysis
========================================
Builds a directed graph of airline routes using NetworkX.
Computes structural metrics that identify M&A vulnerability:

  - Betweenness centrality: hubs that bridge many city-pairs
  - Clustering coefficient: local network redundancy
  - Hub concentration (Herfindahl-like index): monopoly vs. distributed routes
  - Network vulnerability score: low redundancy + high centrality = M&A target
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from src.utils import CONFIG, get_logger

logger = get_logger(__name__)


def build_carrier_graph(routes_df: pd.DataFrame, carrier: str) -> "nx.DiGraph":
    """
    Build a directed NetworkX graph for a single carrier.

    Args:
        routes_df: OpenFlights routes DataFrame [airline, src_airport, dst_airport].
        carrier:   Carrier IATA code (e.g. 'UA').

    Returns:
        nx.DiGraph where nodes = airports, edges = routes.
    """
    if not HAS_NX:
        raise ImportError("networkx is required: pip install networkx")

    sub = routes_df[routes_df["airline"] == carrier]
    G = nx.DiGraph()
    for _, row in sub.iterrows():
        G.add_edge(str(row["src_airport"]), str(row["dst_airport"]))
    return G


def compute_network_metrics(
    routes_df: pd.DataFrame,
    carriers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-carrier network metrics.

    Metrics returned:
      - n_airports:            Number of airports served
      - n_routes:              Total directed routes
      - avg_betweenness:       Average node betweenness (normalised)
      - max_betweenness:       Max node betweenness (hub dependency)
      - avg_clustering:        Average clustering coefficient (undirected)
      - hub_concentration:     HHI-style concentration of departures per airport
      - network_vulnerability: Composite score (0=robust, 1=fragile/target)

    Args:
        routes_df: OpenFlights routes DataFrame.
        carriers:  List of carrier codes to include. None = all in data.

    Returns:
        DataFrame with one row per carrier.
    """
    if not HAS_NX:
        logger.warning("NetworkX not installed — returning synthetic metrics")
        return _synthetic_metrics(carriers or CONFIG["airlines"]["bts_codes"])

    carrier_list = carriers or routes_df["airline"].unique().tolist()
    records = []

    for carrier in carrier_list:
        G = build_carrier_graph(routes_df, carrier)
        if G.number_of_nodes() < 2:
            continue

        G_undirected = G.to_undirected()

        # Betweenness centrality (normalised)
        bc = nx.betweenness_centrality(G, normalized=True)
        bc_vals = list(bc.values())

        # Clustering
        cl = nx.clustering(G_undirected)
        cl_vals = list(cl.values())

        # Hub concentration: how many routes go through top 3 airports?
        out_degree = dict(G.out_degree())
        total_deps = sum(out_degree.values()) or 1
        sorted_deps = sorted(out_degree.values(), reverse=True)
        top3_share = sum(sorted_deps[:3]) / total_deps  # HHI proxy

        # Composite vulnerability:
        #   High betweenness (few key hubs) + low clustering (no redundancy) = fragile
        avg_bc  = float(np.mean(bc_vals)) if bc_vals else 0.0
        max_bc  = float(np.max(bc_vals))  if bc_vals else 0.0
        avg_cl  = float(np.mean(cl_vals)) if cl_vals else 0.0

        vulnerability = float(
            0.4 * max_bc +           # high hub dependency
            0.3 * (1 - avg_cl) +     # low redundancy
            0.3 * top3_share         # concentrated routing
        )

        records.append({
            "carrier":           carrier,
            "n_airports":        G.number_of_nodes(),
            "n_routes":          G.number_of_edges(),
            "avg_betweenness":   round(avg_bc, 5),
            "max_betweenness":   round(max_bc, 5),
            "avg_clustering":    round(avg_cl, 5),
            "hub_concentration": round(top3_share, 4),
            "network_vulnerability": round(vulnerability, 4),
        })

    df = pd.DataFrame(records).sort_values("network_vulnerability", ascending=False)
    return df.reset_index(drop=True)


def _synthetic_metrics(carriers: list[str]) -> pd.DataFrame:
    """Return plausible synthetic network metrics when NetworkX is unavailable."""
    rng = np.random.default_rng(42)
    records = []
    for c in carriers:
        vuln = float(rng.uniform(0.2, 0.8))
        records.append({
            "carrier":           c,
            "n_airports":        int(rng.integers(20, 60)),
            "n_routes":          int(rng.integers(40, 200)),
            "avg_betweenness":   round(float(rng.uniform(0.01, 0.15)), 5),
            "max_betweenness":   round(float(rng.uniform(0.15, 0.55)), 5),
            "avg_clustering":    round(float(rng.uniform(0.1, 0.5)), 5),
            "hub_concentration": round(float(rng.uniform(0.3, 0.7)), 4),
            "network_vulnerability": round(vuln, 4),
        })
    return pd.DataFrame(records).sort_values("network_vulnerability", ascending=False).reset_index(drop=True)


def identify_ma_targets(
    network_metrics: pd.DataFrame,
    overlap_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Combine network vulnerability with route overlap to score M&A targets.

    A carrier is a likely M&A target/acquiree if:
      1. High network vulnerability (fragile hub structure)
      2. High overlap with a stronger carrier (synergistic acquisition)

    Args:
        network_metrics: Output of compute_network_metrics().
        overlap_df:      Output of route_overlap.temporal_overlap() (latest period).
        top_n:           Number of top targets to return.

    Returns:
        DataFrame with M&A target scores.
    """
    latest = overlap_df[overlap_df["period"] == overlap_df["period"].max()].copy()

    # For each carrier, compute its max overlap with any other carrier
    max_overlap = {}
    for carr in network_metrics["carrier"]:
        rows_a = latest[latest["carrier_a"] == carr]["jaccard"]
        rows_b = latest[latest["carrier_b"] == carr]["jaccard"]
        all_overlaps = pd.concat([rows_a, rows_b])
        max_overlap[carr] = float(all_overlaps.max()) if not all_overlaps.empty else 0.0

    nm = network_metrics.copy()
    nm["max_route_overlap"] = nm["carrier"].map(max_overlap).fillna(0)
    nm["ma_pressure_score"] = (
        0.5 * nm["network_vulnerability"] + 0.5 * nm["max_route_overlap"]
    ).round(4)

    return nm.sort_values("ma_pressure_score", ascending=False).head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    from src.ingestion.openflights import load_routes
    from src.features.route_overlap import temporal_overlap
    from src.ingestion.bts_loader import load_bts

    routes = load_routes()
    carriers = CONFIG["airlines"]["bts_codes"]
    metrics = compute_network_metrics(routes, carriers)

    print("=== Network Metrics ===")
    print(metrics.to_string())

    bts = load_bts()
    overlap = temporal_overlap(bts, carriers)
    targets = identify_ma_targets(metrics, overlap)
    print("\n=== Top M&A Targets ===")
    print(targets.to_string())
