# Aviation Alpha

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2942,100:185FA5&height=200&section=header&text=Aviation%20Alpha&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Alternative%20Data%20%7C%20Quant%20Research%20Pipeline&descAlignY=58&descSize=18" />
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" /></a>
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" /></a>
  <a href="https://www.statsmodels.org"><img src="https://img.shields.io/badge/statsmodels-Granger%20Causality-00897B?style=for-the-badge" /></a>
  <a href="https://networkx.org"><img src="https://img.shields.io/badge/NetworkX-Route%20Graphs-orange?style=for-the-badge" /></a>
  <a href="https://www.flightaware.com/aeroapi/"><img src="https://img.shields.io/badge/FlightAware-AeroAPI%20v4-005DA8?style=for-the-badge" /></a>
</p>

<p align="center">
  <strong>An end-to-end quantitative research pipeline that extracts trading signals from aviation stress data.</strong><br/>
  Route competition · Operational disruptions · Granger causality · Live flight data · Backtesting
</p>

---

## 📌 Overview

**Aviation Alpha** transforms public aviation data into actionable trading signals. It models three complementary alpha sources:

| Signal | What it Measures | Why it Works |
|---|---|---|
| 🔀 **Route Overlap** | Jaccard similarity between airline route networks | High overlap historically precedes merger activity (e.g. AA/US Airways 2013) |
| 💥 **Disruption Spikes** | Rolling z-score of cancellation/delay rates | Abnormal operational stress is a leading indicator of financial distress |
| 🛰️ **Network Vulnerability** | Graph centrality & clustering via NetworkX | Low redundancy = fragile carrier = M&A target |

These signals are econometrically validated with **Granger causality tests** and backtested against **SPY** and **XAL** (airline ETF).

---

## 🏗️ Architecture

```
aviation-alpha/
├── src/
│   ├── ingestion/
│   │   ├── bts_loader.py       # BTS T-100 segment data (realistic simulation)
│   │   ├── openflights.py      # OpenFlights routes & airports (live/cached)
│   │   └── flightaware.py      # FlightAware AeroAPI v4 client
│   ├── features/
│   │   ├── route_overlap.py    # Jaccard similarity, temporal windowing
│   │   ├── disruption.py       # Rolling z-score spike detection
│   │   └── network.py          # NetworkX graph metrics & hub vulnerability
│   ├── signals/
│   │   ├── ma_pressure.py      # Composite M&A pressure score
│   │   └── volatility.py       # Airline stock volatility via yfinance
│   ├── econometrics/
│   │   └── granger.py          # Granger causality tests (statsmodels)
│   ├── backtesting/
│   │   └── backtest.py         # Signal → P&L, Sharpe ratio, drawdown
│   └── dashboard/
│       └── app.py              # 6-tab Streamlit dashboard
├── tests/
│   └── test_features.py
├── config.yaml
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Aadi7171/aviation-alpha.git
cd aviation-alpha
pip install -r requirements.txt
```

### 2. Configure API Key (Optional)

```bash
cp .env.example .env
# Add your FlightAware key:
# FLIGHTAWARE_API_KEY=your_key_here
```

> **Free tier:** 500 requests/month — [flightaware.com/aeroapi](https://www.flightaware.com/aeroapi/)
> The pipeline runs **fully offline** without a key using simulated data.

### 3. Launch the Dashboard

```bash
streamlit run src/dashboard/app.py
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

---

## 📊 Methodology

### Signal 1 — Route Overlap (M&A Pressure)

```
Jaccard(A, B) = |routes_A ∩ routes_B| / |routes_A ∪ routes_B|
```

Measures competitive overlap between two carriers. High Jaccard similarity signals route duplication, which historically precedes consolidation activity.

### Signal 2 — Disruption Spikes

```
z = (cancel_rate - rolling_mean) / rolling_std
spike = |z| > 2.5σ
```

A 2.5σ spike in cancellations or delays flags abnormal operational stress — a leading indicator of financial pressure on a carrier.

### Signal 3 — Network Vulnerability

```python
# NetworkX metrics per carrier
betweenness_centrality(G)   # Hub importance
clustering_coefficient(G)   # Route redundancy
```

- **Low clustering** → no redundant routes → fragile network → M&A target
- **High hub concentration** → single point of failure → acquisition risk

### Composite M&A Pressure Score

```
score = 0.35 × network_vulnerability
      + 0.40 × max_route_overlap
      + 0.25 × disruption_spike_rate
```

### Granger Causality Validation

```python
from statsmodels.tsa.stattools import grangercausalitytests
# H₀: X does NOT Granger-cause Y
# Reject H₀ if p < 0.05 → X has predictive power over Y
```

| Cause (X) | Effect (Y) |
|---|---|
| Market Disruption Index | Airline Market Volatility |
| Average Route Overlap | Market Volatility |

---

## 🛰️ Live Data Integration

The `FlightAwareClient` connects to [AeroAPI v4](https://www.flightaware.com/aeroapi/portal/documentation) for real-time signal computation:

```python
from src.ingestion.flightaware import FlightAwareClient

client = FlightAwareClient()  # reads FLIGHTAWARE_API_KEY from .env

# Live disruption scores across major airports
scores = client.compute_live_disruption_scores()

# Departures from a specific airport
flights = client.get_airport_departures("KATL")
```

**API features:**
- ✅ Response caching (TTL: 60 min) — conserves free-tier quota
- ✅ Polite rate-limiting between requests
- ✅ Graceful fallback to realistic simulation when no key is present

---

## 📈 Backtesting

The backtester converts the composite signal into a long/short P&L curve and reports:

- Annualized **Sharpe Ratio**
- **Max Drawdown**
- Signal correlation vs. **SPY** and **XAL**

---

## 🔬 Future Enhancements

| Area | Enhancement |
|---|---|
| Causality Depth | Vector Error Correction Models (VECM) for cointegrated series |
| Feature Engineering | Passenger load factor alpha, cargo volume signals |
| Signal Coverage | International routes (Eurocontrol, IATA datasets) |
| Live Scalability | Webhook-based real-time ingestion instead of polling |
| Risk Management | Kelly criterion position sizing in backtest |
| ML Layer | LSTM / Gradient Boosting on feature stack |

---

## 📚 References

- [BTS T-100 Domestic Segment Data](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FLL)
- [OpenFlights Route Database](https://openflights.org/data.html)
- [FlightAware AeroAPI v4 Docs](https://www.flightaware.com/aeroapi/portal/documentation)
- Granger, C.W.J. (1969). *Investigating Causal Relations by Econometric Models and Cross-spectral Methods*. Econometrica, 37(3).

---

<p align="center">
  Built for quantitative research &nbsp;·&nbsp; Python &nbsp;·&nbsp; statsmodels &nbsp;·&nbsp; NetworkX &nbsp;·&nbsp; yfinance &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Plotly
</p>
