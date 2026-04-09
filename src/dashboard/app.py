"""
Aviation Alpha — Streamlit Interactive Dashboard
==================================================
Run with:  streamlit run src/dashboard/app.py

Tabs:
  1. 📊 Overview          — Project summary & live KPIs
  2. ✈️  Route Overlap    — Jaccard similarity network & heatmap
  3. 💥 Disruption Radar  — Rolling z-score spike timeline
  4. 📈 Signal & Backtest — M&A pressure score + backtested P&L
  5. 🔬 Granger Causality — Econometric test results
  6. 🛰️  Live Feed        — Real-time FlightAware snapshot
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils import CONFIG

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aviation Alpha | Quantitative Research",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a0e1a 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #111827 100%);
    border-right: 1px solid rgba(99, 179, 237, 0.15);
}
.metric-card {
    background: linear-gradient(135deg, rgba(17,24,39,0.9) 0%, rgba(30,41,59,0.9) 100%);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(99, 179, 237, 0.5);
    transform: translateY(-2px);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #63b3ed;
}
.metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    border-left: 3px solid #63b3ed;
    padding-left: 12px;
    margin-bottom: 16px;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-green  { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-red    { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.badge-yellow { background: rgba(251,191,36,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.badge-blue   { background: rgba(99,179,237,0.15);  color: #63b3ed; border: 1px solid rgba(99,179,237,0.3); }
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,27,42,0.6)",
    font=dict(family="Inter", color="#e2e8f0"),
)

CARRIERS_MAP = CONFIG["airlines"]["tickers"]
carriers = CONFIG["airlines"]["bts_codes"]


# ── Data Loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading BTS data…")
def load_all_data():
    from src.ingestion.bts_loader import load_bts, monthly_carrier_summary
    from src.ingestion.openflights import load_routes, carrier_route_sets
    from src.features.route_overlap import temporal_overlap, build_overlap_matrix
    from src.features.disruption import compute_disruption_timeseries, aggregate_market_disruption
    from src.features.network import compute_network_metrics
    from src.signals.ma_pressure import compute_ma_pressure_score
    from src.signals.volatility import get_airline_volatility, compute_vol_disruption_correlation

    bts      = load_bts()
    summary  = monthly_carrier_summary(bts)
    routes   = load_routes()
    crs      = carrier_route_sets(routes)

    overlap      = temporal_overlap(bts, carriers)
    static_matrix = build_overlap_matrix(crs, carriers_of_interest=carriers)
    disruption   = compute_disruption_timeseries(summary)
    market_dis   = aggregate_market_disruption(disruption)
    net_metrics  = compute_network_metrics(routes, carriers)
    vol_df       = get_airline_volatility()
    corr_df      = compute_vol_disruption_correlation(vol_df, market_dis)
    ma_scores    = compute_ma_pressure_score(net_metrics, overlap, disruption, carriers)

    return {
        "bts": bts, "summary": summary, "routes": routes,
        "overlap": overlap, "static_matrix": static_matrix,
        "disruption": disruption, "market_dis": market_dis,
        "net_metrics": net_metrics, "vol_df": vol_df,
        "corr_df": corr_df, "ma_scores": ma_scores,
    }


@st.cache_data(show_spinner="Running Granger tests…", ttl=3600)
def load_granger(market_dis, vol_df, overlap):
    from src.econometrics.granger import run_full_causality_suite
    return run_full_causality_suite(market_dis, vol_df, overlap, carriers)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Aviation Alpha")
    st.markdown("**Quantitative Research Pipeline**")
    st.markdown("---")
    st.markdown("### Configuration")
    sig_threshold = st.slider("M&A Signal Threshold", 0.3, 0.9, SIGNAL_THRESHOLD := CONFIG["backtest"]["signal_threshold"], 0.05)
    holding_days  = st.slider("Holding Period (days)", 10, 90, CONFIG["backtest"]["holding_period_days"], 5)
    st.markdown("---")
    st.markdown("### Data Status")
    st.markdown("🟢 **BTS Data**: Offline (Simulated)")
    st.markdown("🟡 **OpenFlights**: Live / Cached")
    import os
    api_key_set = bool(os.getenv("FLIGHTAWARE_API_KEY", ""))
    if api_key_set:
        st.markdown("🟢 **FlightAware API**: Connected")
    else:
        st.markdown("🔴 **FlightAware API**: Not configured")
        st.markdown(
            "[Get free API key →](https://www.flightaware.com/aeroapi/)",
            unsafe_allow_html=False,
        )
    st.markdown("---")
    st.markdown("*Built with Python · statsmodels · NetworkX · yfinance · Streamlit*")


# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Initialising pipeline…"):
    data = load_all_data()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "✈️ Route Overlap",
    "💥 Disruption Radar",
    "📈 Signal & Backtest",
    "🔬 Granger Causality",
    "🛰️ Live Feed",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("# Aviation Alpha Research Pipeline")
    st.markdown(
        "Exploring **alternative data alpha from aviation stress** — modelling route overlap, "
        "operational disruption, and their relationship to M&A pressure and market volatility."
    )
    st.markdown("---")

    # KPI row
    ma_latest = data["ma_scores"].groupby("carrier")["ma_pressure_score"].last().reset_index()
    top_target = ma_latest.sort_values("ma_pressure_score", ascending=False).iloc[0]
    spikes_total = int(data["disruption"]["spike"].sum())
    avg_overlap  = float(data["overlap"]["jaccard"].mean())
    max_dis      = float(data["market_dis"]["market_disruption_index"].max())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{top_target['carrier']}</div>
            <div class="metric-label">Top M&A Target</div>
            <div style="color:#94a3b8;font-size:0.8rem">Score: {top_target['ma_pressure_score']:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{spikes_total:,}</div>
            <div class="metric-label">Disruption Spikes</div>
            <div style="color:#94a3b8;font-size:0.8rem">Across all carriers & periods</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg_overlap:.3f}</div>
            <div class="metric-label">Avg Route Overlap</div>
            <div style="color:#94a3b8;font-size:0.8rem">Jaccard across all pairs</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{max_dis:.2f}σ</div>
            <div class="metric-label">Peak Market Stress</div>
            <div style="color:#94a3b8;font-size:0.8rem">Max disruption index</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Market disruption timeline
    st.markdown('<div class="section-title">Market Disruption Index — All Carriers</div>', unsafe_allow_html=True)
    fig_mdi = px.area(
        data["market_dis"], x="DATE", y="market_disruption_index",
        color_discrete_sequence=["#63b3ed"],
        labels={"market_disruption_index": "Disruption Index", "DATE": "Date"},
    )
    fig_mdi.add_hline(y=2.5, line_dash="dash", line_color="#f87171", annotation_text="Threshold (2.5σ)")
    fig_mdi.update_layout(**PLOTLY_DARK, height=320)
    st.plotly_chart(fig_mdi, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-title">Network Vulnerability Rankings</div>', unsafe_allow_html=True)
        nm = data["net_metrics"][["carrier", "n_routes", "hub_concentration", "network_vulnerability"]].copy()
        fig_bar = px.bar(
            nm.sort_values("network_vulnerability"),
            x="network_vulnerability", y="carrier", orientation="h",
            color="network_vulnerability",
            color_continuous_scale=["#34d399", "#fbbf24", "#f87171"],
        )
        fig_bar.update_layout(**PLOTLY_DARK, height=320, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Vol / Disruption Rolling Correlation</div>', unsafe_allow_html=True)
        corr_df = data["corr_df"].dropna(subset=["rolling_corr"])
        fig_corr = px.line(
            corr_df, x="DATE", y="rolling_corr",
            color_discrete_sequence=["#a78bfa"],
        )
        fig_corr.add_hline(y=0, line_dash="dot", line_color="#64748b")
        fig_corr.update_layout(**PLOTLY_DARK, height=320)
        st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ROUTE OVERLAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## ✈️ Route Overlap Analysis")
    st.markdown("Jaccard similarity between carrier route networks — high overlap signals consolidation pressure.")

    # Heatmap
    st.markdown('<div class="section-title">Pairwise Jaccard Similarity Matrix</div>', unsafe_allow_html=True)
    mat = data["static_matrix"]
    fig_heat = go.Figure(go.Heatmap(
        z=mat.values, x=mat.columns.tolist(), y=mat.index.tolist(),
        colorscale="Blues", zmin=0, zmax=1,
        text=mat.round(3).values, texttemplate="%{text}",
    ))
    fig_heat.update_layout(**PLOTLY_DARK, height=420)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Temporal overlap
    st.markdown('<div class="section-title">Route Overlap Over Time — Select Pair</div>', unsafe_allow_html=True)
    pairs = data["overlap"]["pair"].unique().tolist()
    selected_pair = st.selectbox("Carrier Pair", pairs, index=0)
    pair_df = data["overlap"][data["overlap"]["pair"] == selected_pair]
    fig_time = px.line(
        pair_df, x="period", y="jaccard",
        markers=True, color_discrete_sequence=["#63b3ed"],
    )
    fig_time.update_layout(**PLOTLY_DARK, height=320)
    st.plotly_chart(fig_time, use_container_width=True)

    # Top pairs table
    st.markdown('<div class="section-title">Top Overlapping Pairs (Latest Period)</div>', unsafe_allow_html=True)
    from src.features.route_overlap import top_overlapping_pairs
    top_pairs = top_overlapping_pairs(data["overlap"], n=10)
    st.dataframe(
        top_pairs.style.background_gradient(cmap="Blues", subset=["jaccard"]),
        use_container_width=True, height=300,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DISRUPTION RADAR
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 💥 Disruption Radar")
    st.markdown("Rolling z-score spike detection on cancellation rates and departure delays.")

    sel_carrier = st.selectbox("Select Carrier", carriers, key="dis_carrier")
    dis_sub = data["disruption"][data["disruption"]["CARRIER"] == sel_carrier].copy()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Disruption Score Timeline</div>', unsafe_allow_html=True)
        fig_dis = go.Figure()
        fig_dis.add_trace(go.Scatter(
            x=dis_sub["DATE"], y=dis_sub["disruption_score"],
            fill="tozeroy", line=dict(color="#63b3ed"), name="Disruption Score",
        ))
        fig_dis.add_trace(go.Scatter(
            x=dis_sub[dis_sub["spike"] == 1]["DATE"],
            y=dis_sub[dis_sub["spike"] == 1]["disruption_score"],
            mode="markers", marker=dict(color="#f87171", size=9, symbol="x"),
            name="Spike Event",
        ))
        fig_dis.add_hline(y=2.5, line_dash="dash", line_color="#fbbf24", annotation_text="2.5σ threshold")
        fig_dis.update_layout(**PLOTLY_DARK, height=350)
        st.plotly_chart(fig_dis, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Cancel Rate & Delay Z-Score</div>', unsafe_allow_html=True)
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(
            x=dis_sub["DATE"], y=dis_sub["cancel_zscore"],
            name="Cancel Z-Score", line=dict(color="#f87171"),
        ))
        fig_z.add_trace(go.Scatter(
            x=dis_sub["DATE"], y=dis_sub["delay_zscore"],
            name="Delay Z-Score", line=dict(color="#fbbf24"),
        ))
        fig_z.add_hline(y=2.5, line_dash="dot", line_color="white", opacity=0.3)
        fig_z.add_hline(y=-2.5, line_dash="dot", line_color="white", opacity=0.3)
        fig_z.update_layout(**PLOTLY_DARK, height=350)
        st.plotly_chart(fig_z, use_container_width=True)

    # Carrier comparison heatmap
    st.markdown('<div class="section-title">Disruption Heatmap — All Carriers</div>', unsafe_allow_html=True)
    pivot = data["disruption"].pivot_table(
        index="CARRIER", columns="DATE", values="disruption_score", aggfunc="mean"
    )
    # Sample columns to avoid overcrowding
    step = max(1, len(pivot.columns) // 30)
    pivot_sampled = pivot.iloc[:, ::step]
    fig_hm = go.Figure(go.Heatmap(
        z=pivot_sampled.values,
        x=[str(c.date()) for c in pivot_sampled.columns],
        y=pivot_sampled.index.tolist(),
        colorscale="RdYlBu_r", zmid=2.5,
    ))
    fig_hm.update_layout(**PLOTLY_DARK, height=320)
    st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SIGNAL & BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📈 M&A Pressure Signal & Backtest")

    # Latest scores
    st.markdown('<div class="section-title">Latest M&A Pressure Scores</div>', unsafe_allow_html=True)
    latest_period = data["ma_scores"]["period"].max()
    latest_scores = data["ma_scores"][data["ma_scores"]["period"] == latest_period].copy()

    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        fig_scores = px.bar(
            latest_scores.sort_values("ma_pressure_score"),
            x="ma_pressure_score", y="carrier", orientation="h",
            color="ma_pressure_score",
            color_continuous_scale=["#34d399", "#fbbf24", "#f87171"],
            text="ma_pressure_score",
        )
        fig_scores.add_vline(x=sig_threshold, line_dash="dash", line_color="#63b3ed",
                              annotation_text=f"Threshold ({sig_threshold})")
        fig_scores.update_layout(**PLOTLY_DARK, height=380)
        st.plotly_chart(fig_scores, use_container_width=True)

    with col_r:
        st.markdown(f"**Period:** `{latest_period}`")
        for _, row in latest_scores.sort_values("ma_pressure_score", ascending=False).iterrows():
            badge_cls = "badge-red" if row["ma_pressure_score"] > sig_threshold else "badge-green"
            st.markdown(
                f"<span class='badge {badge_cls}'>{row['carrier']}</span> "
                f"Score: **{row['ma_pressure_score']:.4f}** | "
                f"Overlap: {row['max_route_overlap']:.3f} | "
                f"Vuln: {row['network_vuln']:.3f}",
                unsafe_allow_html=True,
            )

    # Backtest
    st.markdown("---")
    st.markdown('<div class="section-title">Backtested Signal P&L</div>', unsafe_allow_html=True)

    if st.button("▶ Run Backtest", type="primary"):
        with st.spinner("Running backtest…"):
            from src.backtesting.backtest import run_backtest
            from src.signals.volatility import _download_prices, _synthetic_prices

            tickers = list(CARRIERS_MAP.keys())
            prices = _download_prices(tickers, str(CONFIG["data"]["start_year"]), str(CONFIG["data"]["end_year"]))
            if prices is None:
                prices = _synthetic_prices(tickers, f"{CONFIG['data']['start_year']}-01-01", f"{CONFIG['data']['end_year']}-12-31")

            result = run_backtest(data["ma_scores"], prices, threshold=sig_threshold, holding_days=holding_days)

        # KPI row
        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        metrics = [
            ("Total Return", f"{result['total_return']}%"),
            ("Sharpe Ratio", str(result['sharpe'])),
            ("Max Drawdown", f"{result['max_drawdown']}%"),
            ("Hit Rate",     f"{result['hit_rate']}%"),
            ("# Trades",     str(result['n_trades'])),
        ]
        for col, (label, val) in zip([kc1, kc2, kc3, kc4, kc5], metrics):
            with col:
                st.metric(label, val)

        # Equity curve
        if not result["equity_curve"].empty and len(result["equity_curve"]) > 1:
            eq = result["equity_curve"].reset_index()
            eq.columns = ["Date", "Equity"]
            fig_eq = px.line(eq, x="Date", y="Equity", color_discrete_sequence=["#34d399"])
            fig_eq.update_layout(**PLOTLY_DARK, height=300)
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("No trades generated with current threshold. Try lowering the signal threshold in the sidebar.")

        if not result["trades"].empty:
            st.markdown('<div class="section-title">Trade Log</div>', unsafe_allow_html=True)
            st.dataframe(result["trades"], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GRANGER CAUSALITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🔬 Granger Causality Tests")
    st.markdown(
        "Statistical tests to determine if aviation stress signals have **predictive causality** "
        "over market volatility. Methodology: `grangercausalitytests` (statsmodels) with AIC lag selection."
    )

    if st.button("▶ Run Granger Tests", type="primary"):
        with st.spinner("Running econometric tests (this may take ~30s)…"):
            granger_results = load_granger(data["market_dis"], data["vol_df"], data["overlap"])

        st.markdown('<div class="section-title">Test Results Summary</div>', unsafe_allow_html=True)
        for _, row in granger_results.iterrows():
            sig_badge = '<span class="badge badge-green">✅ Significant</span>' if "Yes" in str(row.get("Significant", "")) else '<span class="badge badge-red">❌ Not Significant</span>'
            st.markdown(
                f"**{row['Test']}** &nbsp; {sig_badge} &nbsp; "
                f"Min p-value: `{row['Min P-Value']}` &nbsp; Optimal lag: `{row['Optimal Lag']}`",
                unsafe_allow_html=True,
            )

        st.dataframe(granger_results, use_container_width=True)

        st.markdown("---")
        st.markdown("""
        **Interpretation guide:**
        - **p < 0.05** → Reject H₀ → The cause variable has statistically significant predictive power over the effect variable
        - **Optimal Lag** → How many months back the signal leads the market response
        - **H₀**: Aviation stress does NOT Granger-cause volatility/M&A activity
        """)
    else:
        st.info("Click 'Run Granger Tests' to execute the econometric causality analysis.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — LIVE FEED
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 🛰️ Live FlightAware Feed")

    import os
    if not os.getenv("FLIGHTAWARE_API_KEY") or os.getenv("FLIGHTAWARE_API_KEY") == "your_api_key_here":
        st.warning(
            "⚠️ **FlightAware API key not configured.** "
            "Showing simulated live data. "
            "Set `FLIGHTAWARE_API_KEY` in your `.env` file to connect to live data.\n\n"
            "[Get a free key at FlightAware →](https://www.flightaware.com/aeroapi/)"
        )

    if st.button("🔄 Fetch Live Snapshot", type="primary"):
        with st.spinner("Fetching flight data…"):
            from src.ingestion.flightaware import FlightAwareClient
            client = FlightAwareClient()
            live_scores = client.compute_live_disruption_scores()
            katl_flights = client.get_airport_departures("KATL")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="section-title">Live Disruption Scores by Carrier</div>', unsafe_allow_html=True)
            if not live_scores.empty:
                fig_live = px.bar(
                    live_scores.sort_values("disruption_score"),
                    x="disruption_score", y="carrier", orientation="h",
                    color="disruption_score",
                    color_continuous_scale=["#34d399", "#fbbf24", "#f87171"],
                )
                fig_live.update_layout(**PLOTLY_DARK, height=350)
                st.plotly_chart(fig_live, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-title">KATL Departure Feed</div>', unsafe_allow_html=True)
            if not katl_flights.empty:
                display_cols = ["ident", "carrier", "destination", "status", "departure_delay"]
                st.dataframe(
                    katl_flights[display_cols].head(20),
                    use_container_width=True, height=350,
                )

        st.markdown('<div class="section-title">Full Live Disruption Scores</div>', unsafe_allow_html=True)
        st.dataframe(live_scores, use_container_width=True)
    else:
        st.info("Click 'Fetch Live Snapshot' to pull current flight data.")
