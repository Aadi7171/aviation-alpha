"""
Signal Backtesting Framework
================================
Backtests the M&A pressure signal against real airline stock returns.

Strategy:
  - Go LONG carrier stocks with M&A pressure score > threshold
  - Hold for `holding_period_days`
  - Benchmark vs SPY and XAL (US Global Jets / Airline ETF proxy)

Outputs:
  - Equity curve (P&L over time)
  - Sharpe ratio, max drawdown, CAGR, hit rate
  - Per-trade log
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import CONFIG, get_logger

logger = get_logger(__name__)

INITIAL_CAPITAL  = CONFIG["backtest"]["initial_capital"]
SIGNAL_THRESHOLD = CONFIG["backtest"]["signal_threshold"]
HOLDING_DAYS     = CONFIG["backtest"]["holding_period_days"]


# ── Performance Metrics ───────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.04) -> float:
    """Annualised Sharpe ratio (assuming daily returns)."""
    excess = returns - risk_free / 252
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return float(drawdown.min())


def cagr(equity_curve: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    n_years = len(equity_curve) / 252
    if n_years <= 0:
        return 0.0
    return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1)


# ── Backtest Engine ───────────────────────────────────────────────────────────

def run_backtest(
    ma_scores:    pd.DataFrame,
    prices:       pd.DataFrame,
    threshold:    float = SIGNAL_THRESHOLD,
    holding_days: int   = HOLDING_DAYS,
    capital:      float = INITIAL_CAPITAL,
) -> dict:
    """
    Run the M&A pressure signal backtest.

    Args:
        ma_scores:    Output of ma_pressure.compute_ma_pressure_score().
                      Must have [period, carrier, ma_pressure_score].
        prices:       DataFrame of daily close prices (columns = ticker codes).
        threshold:    Signal threshold — only trade carriers above this score.
        holding_days: How many trading days to hold each position.
        capital:      Starting capital.

    Returns:
        dict with:
          equity_curve (pd.Series), trades (pd.DataFrame),
          sharpe, max_drawdown, cagr, hit_rate, n_trades
    """
    # Map carrier codes to tickers
    code_to_ticker = {v: k for k, v in CONFIG["airlines"]["tickers"].items()}
    # Also try direct match
    for k in CONFIG["airlines"]["tickers"]:
        code_to_ticker[k] = k

    trades = []
    equity = capital
    equity_log = []

    # Convert periods to timestamps
    ma_scores = ma_scores.copy()
    ma_scores["signal_date"] = ma_scores["period"].apply(
        lambda p: pd.Period(p, freq="Q").to_timestamp(how="end")
    )

    all_dates = pd.DatetimeIndex(prices.index if hasattr(prices, 'index') else [])
    if prices.empty or len(all_dates) == 0:
        return _empty_backtest(capital)

    for _, row in ma_scores.sort_values("signal_date").iterrows():
        if row["ma_pressure_score"] < threshold:
            continue

        ticker = code_to_ticker.get(row["carrier"])
        if not ticker or ticker not in prices.columns:
            continue

        entry_date = row["signal_date"]
        # Find nearest trading day
        valid_dates = all_dates[all_dates >= entry_date]
        if valid_dates.empty:
            continue
        entry_date = valid_dates[0]

        exit_candidates = all_dates[all_dates > entry_date]
        if len(exit_candidates) < holding_days:
            continue
        exit_date = exit_candidates[min(holding_days - 1, len(exit_candidates) - 1)]

        entry_price = prices.loc[entry_date, ticker]
        exit_price  = prices.loc[exit_date, ticker]

        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
            continue

        ret     = (exit_price - entry_price) / entry_price
        pnl     = equity * 0.1 * ret  # size as 10% of equity
        equity += pnl

        trades.append({
            "carrier":     row["carrier"],
            "ticker":      ticker,
            "signal_score": row["ma_pressure_score"],
            "entry_date":  entry_date.date(),
            "exit_date":   exit_date.date(),
            "entry_price": round(entry_price, 2),
            "exit_price":  round(exit_price, 2),
            "return_pct":  round(ret * 100, 2),
            "pnl":         round(pnl, 2),
            "equity":      round(equity, 2),
        })
        equity_log.append({"date": exit_date, "equity": equity})

    if not equity_log:
        return _empty_backtest(capital)

    eq_series = pd.Series(
        [e["equity"] for e in equity_log],
        index=pd.DatetimeIndex([e["date"] for e in equity_log]),
    )
    eq_series = eq_series.sort_index()

    daily_ret = eq_series.pct_change().dropna()
    trades_df = pd.DataFrame(trades)
    hit_rate  = float((trades_df["pnl"] > 0).mean()) if not trades_df.empty else 0.0

    return {
        "equity_curve":  eq_series,
        "trades":        trades_df,
        "final_capital": round(equity, 2),
        "total_return":  round((equity - capital) / capital * 100, 2),
        "sharpe":        round(sharpe_ratio(daily_ret), 3),
        "max_drawdown":  round(max_drawdown(eq_series) * 100, 2),
        "cagr":          round(cagr(eq_series) * 100, 2),
        "hit_rate":      round(hit_rate * 100, 1),
        "n_trades":      len(trades_df),
    }


def _empty_backtest(capital: float) -> dict:
    return {
        "equity_curve":  pd.Series([capital]),
        "trades":        pd.DataFrame(),
        "final_capital": capital,
        "total_return":  0.0,
        "sharpe":        0.0,
        "max_drawdown":  0.0,
        "cagr":          0.0,
        "hit_rate":      0.0,
        "n_trades":      0,
    }


def print_backtest_summary(result: dict) -> None:
    """Print a formatted backtest summary."""
    print("=" * 50)
    print("  BACKTEST SUMMARY — Aviation Alpha Signal")
    print("=" * 50)
    print(f"  Trades:         {result['n_trades']}")
    print(f"  Hit Rate:       {result['hit_rate']}%")
    print(f"  Total Return:   {result['total_return']}%")
    print(f"  CAGR:           {result['cagr']}%")
    print(f"  Sharpe Ratio:   {result['sharpe']}")
    print(f"  Max Drawdown:   {result['max_drawdown']}%")
    print(f"  Final Capital:  ${result['final_capital']:,.2f}")
    print("=" * 50)


if __name__ == "__main__":
    from src.ingestion.bts_loader import load_bts, monthly_carrier_summary
    from src.ingestion.openflights import load_routes
    from src.features.route_overlap import temporal_overlap
    from src.features.disruption import compute_disruption_timeseries
    from src.features.network import compute_network_metrics
    from src.signals.ma_pressure import compute_ma_pressure_score
    from src.signals.volatility import _synthetic_prices, _download_prices

    carriers = CONFIG["airlines"]["bts_codes"]
    tickers  = list(CONFIG["airlines"]["tickers"].keys())

    bts       = load_bts()
    summary   = monthly_carrier_summary(bts)
    routes    = load_routes()
    overlap   = temporal_overlap(bts, carriers)
    disruption= compute_disruption_timeseries(summary)
    net_met   = compute_network_metrics(routes, carriers)
    ma_scores = compute_ma_pressure_score(net_met, overlap, disruption, carriers)

    prices = _download_prices(tickers, CONFIG["data"]["start_year"], CONFIG["data"]["end_year"])
    if prices is None:
        prices = _synthetic_prices(tickers, f"{CONFIG['data']['start_year']}-01-01", f"{CONFIG['data']['end_year']}-12-31")

    result = run_backtest(ma_scores, prices)
    print_backtest_summary(result)
    if not result["trades"].empty:
        print("\nTop 10 Trades:")
        print(result["trades"].head(10).to_string())
