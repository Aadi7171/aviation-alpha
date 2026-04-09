<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2942,100:185FA5&height=200&section=header&text=Aviation%20Alpha&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Alternative%20Data%20%7C%20Quant%20Research%20Pipeline&descAlignY=58&descSize=18" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![statsmodels](https://img.shields.io/badge/Granger-Causality-00897B?style=for-the-badge)](https://www.statsmodels.org)
[![NetworkX](https://img.shields.io/badge/NetworkX-Route%20Graphs-orange?style=for-the-badge)](https://networkx.org)
[![FlightAware](https://img.shields.io/badge/FlightAware-AeroAPI%20v4-005DA8?style=for-the-badge)](https://www.flightaware.com/aeroapi/)

</div>

---

## 📌 Overview

**Aviation Alpha** is an end-to-end quantitative research pipeline exploring **alternative data alpha from aviation stress**. It analyses public aviation datasets to generate trading signals around:

- 🔀 **Route Overlap** → consolidation pressure between airlines
- 💥 **Operational Disruption** → anomalous spikes in cancellations/delays
- 🔬 **Granger Causality** → econometric validation of signal predictive power
- 🛰️ **Live Data** → real-time signal via FlightAware AeroAPI v4
- 📈 **Backtesting** → signal P&L vs SPY and XAL (airline ETF)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Configure FlightAware API

```bash
cp .env.example .env
# Edit .env and add your key:
# FLIGHTAWARE_API_KEY=your_key_here
```

> Free tier: 500 req/month at [flightaware.com/aeroapi](https://www.flightaware.com/aeroapi/)
> The pipeline runs fully offline without a key using simulated data.

### 3. Launch the Dashboard

```bash
streamlit run src/dashboard/app.py
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

---

## 🏗️ Project Structure

```
aviation-alpha/
├── src/
│   ├── ingestion/
│   │   ├── bts_loader.py       # BTS T-100 data (realistic simulation)
│   │   ├── openflights.py      # OpenFlights routes & airports (live/cached)
│   │   └── flightaware.py      # FlightAware AeroAPI v4 client
│   ├── features/
│   │   ├── route_overlap.py    # Jaccard similarity, temporal windowing
│   │   ├── disruption.py       # Rolling z-score spike detection
│   │   └── network.py          # NetworkX graph metrics & vulnerability
│   ├── signals/
│   │   ├── ma_pressure.py      # Composite M&A pressure score
│   │   └── volatility.py       # Airline stock volatility via yfinance
│   ├── econometrics/
│   │   └── granger.py          # Granger causality tests (statsmodels)
│   ├── backtesting/
│   │   └── backtest.py         # Signal → P&L, Sharpe, drawdown
│   └── dashboard/
│       └── app.py              # 6-tab Streamlit dashboard
├── tests/
│   └── test_features.py
├── config.yaml
├── requirements.txt
└── .env.example
```

---

## 📊 Methodology

### Signal 1: Route Overlap (M&A Pressure)

```
Jaccard(A, B) = |routes_A ∩ routes_B| / |routes_A ∪ routes_B|
```

High Jaccard similarity between two carriers signals route competition, which historically precedes acquisition activity (e.g., AA/US Airways overlap before 2013 merger).

### Signal 2: Disruption Spikes

```
z = (cancel_rate - rolling_mean) / rolling_std
spike = |z| > 2.5σ
```

Abnormal cancellation or delay spikes may signal operational distress — a leading indicator of financial pressure.

### Signal 3: Network Vulnerability

Using NetworkX to compute betweenness centrality and clustering coefficients:
- **Low clustering** = no redundant routes = fragile carrier
- **High hub concentration** = single point of failure = M&A target

### Granger Causality

```python
from statsmodels.tsa.stattools import grangercausalitytests
# H0: X does NOT Granger-cause Y
# Reject H0 if p < 0.05 → X has predictive power over Y
```

Tests run:
| Cause | Effect |
|---|---|
| Market Disruption Index | Airline Market Volatility |
| Avg Route Overlap | Market Volatility |

### Composite M&A Pressure Score

```
score = 0.35 × network_vulnerability
      + 0.40 × max_route_overlap
      + 0.25 × disruption_spike_rate
```

---

## 🛰️ Live API Integration

The `FlightAwareClient` in `src/ingestion/flightaware.py` connects to [AeroAPI v4](https://www.flightaware.com/aeroapi/):

```python
from src.ingestion.flightaware import FlightAwareClient

client = FlightAwareClient()  # reads FLIGHTAWARE_API_KEY from .env

# Get live disruption scores across major airports
scores = client.compute_live_disruption_scores()

# Pull departures from a specific airport
flights = client.get_airport_departures("KATL")
```

Features:
- ✅ Response caching (TTL: 60 min) — conserves API quota
- ✅ Rate-limiting with polite delays between requests
- ✅ Graceful fallback to realistic simulation when no key present

---

## 🔬 Areas for Future Enhancement

| Area | Enhancement |
|---|---|
| **Causality Depth** | Vector Error Correction Models (VECM) for cointegrated series |
| **Feature Engineering** | Passenger load factor alpha, cargo volume signals |
| **Signal Coverage** | International routes (Eurocontrol, IATA datasets) |
| **Live Scalability** | Webhook-based real-time ingestion vs. polling |
| **Risk Management** | Kelly criterion position sizing in backtest |
| **ML Layer** | LSTM / Gradient Boosting on feature stack for signal enhancement |

---

## 📚 References

- [BTS T-100 Domestic Segment Data](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FLL)
- [OpenFlights Route Database](https://openflights.org/data.html)
- [FlightAware AeroAPI v4 Docs](https://www.flightaware.com/aeroapi/portal/documentation)
- Granger, C.W.J. (1969). *Investigating Causal Relations by Econometric Models*. Econometrica.

---

<div align="center">

Built for quantitative research · Python · statsmodels · NetworkX · yfinance · Streamlit · Plotly

</div>
