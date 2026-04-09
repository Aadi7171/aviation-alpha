"""
Granger Causality Testing
============================
Tests whether aviation stress signals have predictive (causal) power
over airline stock volatility and M&A activity.

Two key hypotheses:
  H1: Disruption spikes Granger-cause airline stock volatility
  H2: Route overlap changes Granger-cause M&A announcements (proxied by vol)

Method: statsmodels grangercausalitytests
  - Null hypothesis: X does NOT Granger-cause Y
  - Reject H0 (p < 0.05) → X has predictive power over Y
  - Lag selection via AIC
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from src.utils import CONFIG, get_logger

logger = get_logger(__name__)

SIGNIFICANCE = CONFIG["granger"]["significance_level"]
MAX_LAGS     = CONFIG["granger"]["max_lags"]


# ── Stationarity ──────────────────────────────────────────────────────────────

def make_stationary(series: pd.Series, max_diffs: int = 2) -> tuple[pd.Series, int]:
    """
    Difference a series until ADF test rejects unit root.

    Returns:
        (stationary_series, n_diffs_applied)
    """
    s = series.dropna().copy()
    for d in range(max_diffs + 1):
        adf_result = adfuller(s, autolag="AIC")
        p_val = adf_result[1]
        if p_val < SIGNIFICANCE:
            logger.debug("Series stationary after %d diff(s), p=%.4f", d, p_val)
            return s, d
        if d < max_diffs:
            s = s.diff().dropna()
    logger.warning("Series may not be stationary after %d diffs", max_diffs)
    return s, max_diffs


# ── Lag Selection ─────────────────────────────────────────────────────────────

def select_optimal_lag(x: pd.Series, y: pd.Series, max_lags: int = MAX_LAGS) -> int:
    """
    Select optimal lag via AIC from bivariate VAR.

    Simple grid search: test Granger at each lag, pick lowest AIC-equivalent.
    """
    from statsmodels.tsa.vector_ar.var_model import VAR

    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]

    try:
        model = VAR(df)
        result = model.select_order(max_lags)
        optimal = result.aic
        if optimal is None or optimal < 1:
            return 2
        return int(optimal)
    except Exception:
        return 2


# ── Core Granger Test ─────────────────────────────────────────────────────────

def run_granger_test(
    cause: pd.Series,
    effect: pd.Series,
    name: str = "",
    max_lags: int = MAX_LAGS,
    verbose: bool = True,
) -> dict:
    """
    Run Granger causality test: does `cause` Granger-cause `effect`?

    Args:
        cause:    The candidate causal series (X).
        effect:   The series to be explained (Y).
        name:     Label for logging.
        max_lags: Maximum lags to test.
        verbose:  Print results.

    Returns:
        dict with keys: name, min_p_value, optimal_lag, is_significant,
                        all_lags (list of {lag, f_stat, p_value}).
    """
    # Align, stationarise
    cause_s,  _ = make_stationary(cause.copy())
    effect_s, _ = make_stationary(effect.copy())

    aligned = pd.concat([effect_s, cause_s], axis=1).dropna()
    if len(aligned) < max_lags * 3:
        logger.warning("[%s] Insufficient data for Granger test (%d rows)", name, len(aligned))
        return {"name": name, "min_p_value": np.nan, "is_significant": False}

    aligned.columns = ["effect", "cause"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            results = grangercausalitytests(
                aligned[["effect", "cause"]], maxlag=max_lags, verbose=False
            )
        except Exception as exc:
            logger.error("[%s] Granger test failed: %s", name, exc)
            return {"name": name, "min_p_value": np.nan, "is_significant": False}

    all_lags = []
    for lag, res_dict in results.items():
        f_test = res_dict[0].get("ssr_ftest", (np.nan, np.nan))
        f_stat = f_test[0]
        p_val  = f_test[1]
        all_lags.append({"lag": lag, "f_stat": round(f_stat, 4), "p_value": round(p_val, 4)})

    min_p  = min(r["p_value"] for r in all_lags if not np.isnan(r["p_value"]))
    opt_lag = min(all_lags, key=lambda r: r["p_value"])["lag"]

    is_sig = min_p < SIGNIFICANCE

    if verbose:
        status = "✅ SIGNIFICANT" if is_sig else "❌ NOT significant"
        logger.info("[%s] %s  min_p=%.4f  best_lag=%d", name, status, min_p, opt_lag)

    return {
        "name":         name,
        "min_p_value":  round(min_p, 4),
        "optimal_lag":  opt_lag,
        "is_significant": is_sig,
        "all_lags":     all_lags,
    }


# ── Multi-hypothesis Test Suite ───────────────────────────────────────────────

def run_full_causality_suite(
    disruption_df:  pd.DataFrame,
    volatility_df:  pd.DataFrame,
    overlap_df:     pd.DataFrame,
    carriers:       list[str],
) -> pd.DataFrame:
    """
    Run the full suite of Granger causality tests.

    Tests:
      1. Market disruption index → VIX / market volatility
      2. Per-carrier disruption  → per-carrier realized vol
      3. Route overlap (market avg) → market vol

    Args:
        disruption_df:  Output of disruption.aggregate_market_disruption().
        volatility_df:  Output of volatility.compute_realized_volatility() — market level.
        overlap_df:     Output of route_overlap.temporal_overlap().
        carriers:       List of carrier codes.

    Returns:
        Summary DataFrame with test results.
    """
    from src.features.disruption import aggregate_market_disruption

    results = []

    # ── Test 1: Market disruption → market vol ────────────────────────────────
    disruption_df["DATE"] = pd.to_datetime(disruption_df["DATE"])
    volatility_df["Date"] = pd.to_datetime(volatility_df["Date"])

    market_dis = disruption_df.set_index("DATE")["market_disruption_index"]
    market_vol = (
        volatility_df.set_index("Date")
        .get("market_vol", volatility_df.set_index("Date").iloc[:, 0])
    )

    # Resample to monthly for alignment
    market_dis_m = market_dis.resample("ME").mean()
    market_vol_m = market_vol.resample("ME").mean()

    r = run_granger_test(
        cause=market_dis_m,
        effect=market_vol_m,
        name="Market Disruption → Market Vol",
    )
    results.append(r)

    # ── Test 2: Route overlap → market vol ───────────────────────────────────
    if not overlap_df.empty:
        avg_overlap = (
            overlap_df.groupby("period")["jaccard"]
            .mean()
            .reset_index()
        )
        avg_overlap["period"] = avg_overlap["period"].apply(
            lambda p: pd.Period(p, freq="Q").to_timestamp(how="end")
        )
        avg_overlap = avg_overlap.set_index("period")["jaccard"]
        avg_overlap.index = pd.DatetimeIndex(avg_overlap.index).to_period("M").to_timestamp()

        r2 = run_granger_test(
            cause=avg_overlap,
            effect=market_vol_m,
            name="Avg Route Overlap → Market Vol",
        )
        results.append(r2)

    # ── Summary ───────────────────────────────────────────────────────────────
    rows = []
    for r in results:
        rows.append({
            "Test":          r.get("name", "—"),
            "Min P-Value":   r.get("min_p_value", np.nan),
            "Optimal Lag":   r.get("optimal_lag", "—"),
            "Significant":   "✅ Yes" if r.get("is_significant") else "❌ No",
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.ingestion.bts_loader import load_bts, monthly_carrier_summary
    from src.features.disruption import (
        compute_disruption_timeseries, aggregate_market_disruption,
    )
    from src.features.route_overlap import temporal_overlap
    from src.signals.volatility import get_airline_volatility

    carriers = CONFIG["airlines"]["bts_codes"]

    bts = load_bts()
    summary = monthly_carrier_summary(bts)
    disruption = compute_disruption_timeseries(summary)
    market_dis = aggregate_market_disruption(disruption)

    overlap = temporal_overlap(bts, carriers)
    vol_df  = get_airline_volatility()

    results = run_full_causality_suite(market_dis, vol_df, overlap, carriers)
    print("=== Granger Causality Results ===")
    print(results.to_string(index=False))
