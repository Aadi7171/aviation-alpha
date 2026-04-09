"""
Test suite for Aviation Alpha features.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.utils import CONFIG


# ── BTS Loader ────────────────────────────────────────────────────────────────

def test_bts_loader_returns_dataframe():
    from src.ingestion.bts_loader import load_bts
    df = load_bts()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "CARRIER" in df.columns
    assert "cancel_rate" in df.columns


def test_bts_monthly_summary():
    from src.ingestion.bts_loader import load_bts, monthly_carrier_summary
    bts = load_bts()
    summary = monthly_carrier_summary(bts)
    assert "cancel_rate" in summary.columns
    assert "load_factor" in summary.columns
    assert summary["cancel_rate"].between(0, 1).all()


# ── Route Overlap ─────────────────────────────────────────────────────────────

def test_jaccard_similarity():
    from src.features.route_overlap import jaccard_similarity
    a = {("JFK", "LAX"), ("ORD", "DFW"), ("ATL", "MIA")}
    b = {("JFK", "LAX"), ("ORD", "DFW"), ("BOS", "SEA")}
    j = jaccard_similarity(a, b)
    assert 0 <= j <= 1
    assert abs(j - 2/4) < 1e-9  # 2 common / 4 total

def test_jaccard_empty_sets():
    from src.features.route_overlap import jaccard_similarity
    assert jaccard_similarity(set(), set()) == 0.0
    assert jaccard_similarity({("A", "B")}, set()) == 0.0

def test_overlap_matrix_shape():
    from src.ingestion.openflights import load_routes, carrier_route_sets
    from src.features.route_overlap import build_overlap_matrix
    routes = load_routes()
    crs = carrier_route_sets(routes)
    carriers = CONFIG["airlines"]["bts_codes"]
    mat = build_overlap_matrix(crs, carriers_of_interest=carriers)
    assert mat.shape[0] == mat.shape[1]
    # Diagonal should be 1.0
    for i in range(len(mat)):
        assert abs(mat.iloc[i, i] - 1.0) < 1e-9


# ── Disruption ────────────────────────────────────────────────────────────────

def test_disruption_timeseries():
    from src.ingestion.bts_loader import load_bts, monthly_carrier_summary
    from src.features.disruption import compute_disruption_timeseries
    bts = load_bts()
    summary = monthly_carrier_summary(bts)
    disruption = compute_disruption_timeseries(summary)
    assert "disruption_score" in disruption.columns
    assert "spike" in disruption.columns
    assert disruption["spike"].isin([0, 1]).all()

def test_market_disruption_aggregation():
    from src.ingestion.bts_loader import load_bts, monthly_carrier_summary
    from src.features.disruption import compute_disruption_timeseries, aggregate_market_disruption
    bts = load_bts()
    summary = monthly_carrier_summary(bts)
    dis = compute_disruption_timeseries(summary)
    market = aggregate_market_disruption(dis)
    assert "market_disruption_index" in market.columns
    assert "n_carriers_spiking" in market.columns
    assert len(market) > 0


# ── Network Metrics ───────────────────────────────────────────────────────────

def test_network_metrics():
    from src.ingestion.openflights import load_routes
    from src.features.network import compute_network_metrics
    routes = load_routes()
    carriers = CONFIG["airlines"]["bts_codes"]
    metrics = compute_network_metrics(routes, carriers)
    assert "network_vulnerability" in metrics.columns
    assert metrics["network_vulnerability"].between(0, 1).all()


# ── M&A Pressure ─────────────────────────────────────────────────────────────

def test_ma_pressure_score():
    from src.ingestion.bts_loader import load_bts, monthly_carrier_summary
    from src.ingestion.openflights import load_routes
    from src.features.route_overlap import temporal_overlap
    from src.features.disruption import compute_disruption_timeseries
    from src.features.network import compute_network_metrics
    from src.signals.ma_pressure import compute_ma_pressure_score

    carriers = CONFIG["airlines"]["bts_codes"]
    bts      = load_bts()
    summary  = monthly_carrier_summary(bts)
    routes   = load_routes()
    overlap  = temporal_overlap(bts, carriers)
    dis      = compute_disruption_timeseries(summary)
    net_m    = compute_network_metrics(routes, carriers)
    scores   = compute_ma_pressure_score(net_m, overlap, dis, carriers)

    assert "ma_pressure_score" in scores.columns
    assert len(scores) > 0
    assert scores["ma_pressure_score"].between(0, 1).all()


# ── FlightAware (offline / simulated) ────────────────────────────────────────

def test_flightaware_simulated():
    from src.ingestion.flightaware import FlightAwareClient
    client = FlightAwareClient(api_key="")  # Force simulation
    df = client.get_airport_departures("KATL")
    assert isinstance(df, pd.DataFrame)
    assert "carrier" in df.columns
    assert "status" in df.columns

def test_live_disruption_scores():
    from src.ingestion.flightaware import FlightAwareClient
    client = FlightAwareClient(api_key="")
    scores = client.compute_live_disruption_scores()
    assert isinstance(scores, pd.DataFrame)
    if not scores.empty:
        assert "disruption_score" in scores.columns
        assert scores["disruption_score"].between(0, 1).all()
