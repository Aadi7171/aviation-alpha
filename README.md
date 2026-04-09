<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2942,100:185FA5&height=200&section=header&text=Aviation%20Alpha&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Alternative%20Data%20%7C%20Quant%20Research%20Pipeline&descAlignY=58&descSize=18" width="100%"/>

**Aviation Alpha** is a production-grade quantitative research pipeline designed to extract alpha from aviation-based alternative data. It bridges the gap between raw aerospace activity and actionable market intelligence using econometric causality and network graph theory.

![Analysis](https://img.shields.io/badge/Analyst-Aadi7171-blueviolet)
![Backend](https://img.shields.io/badge/Backend-Python--Modular-3776AB)
![Econometrics](https://img.shields.io/badge/Econometrics-Statsmodels-00897B)
![Visuals](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

</div>

---

## 📌 Overview

Aviation Alpha transforms public aviation data into actionable trading signals. It explores three complementary alpha sources validated by econometric proof:

- 🔀 **Route Overlap (M&A Pressure)**: Detects consolidation pressure and competitive overlap between carriers.
- 💥 **Operational Disruption Spikes**: Identifies leading indicators of financial distress via anomalous cancellation/delay patterns.
- 🔬 **Granger Causality**: Statistically proves the predictive lead-lag relationship between aviation stress and market volatility.
- 🛰️ **Live Ingestion**: Real-time signal computation via FlightAware AeroAPI v4 integration.

---

## ✨ Key Features

- **Live Market Radar**: Real-time airport snapshots and carrier disruption scoring.
- **Route Overlap Engine**: Calculates pairwise Jaccard similarity across global route networks.
- **Network Vulnerability Scoring**: Models hub concentration and clustering coefficients using NetworkX.
- **Backtesting Engine**: Event-driven P&L simulation for signal-based strategies vs. SPY/XAL.
- **Premium Research Dashboard**: 6-tab interactive workspace built with Plotly and Streamlit.

---

## 🏗️ Project Structure

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

## 📊 Methodology

### Signal 1 — Route Overlap (M&A Pressure)
```
Jaccard(A, B) = |routes_A ∩ routes_B| / |routes_A ∪ routes_B|
```
High Jaccard similarity between two carriers signals route competition, which historically precedes acquisition activity (e.g. AA/US Airways overlap before 2013 merger).

### Signal 2 — Disruption Spikes
```
z = (cancel_rate - rolling_mean) / rolling_std
spike = |z| > 2.5σ
```
Abnormal cancellation or delay spikes flag operational distress — a leading indicator of financial pressure.

### Signal 3 — Network Vulnerability
Using **NetworkX** to compute structural fragility:
- **Low clustering** = no redundant routes = fragile network = M&A target
- **High hub concentration** = single point of failure = acquisition risk

### Econometric Validation (Granger Causality)
```python
from statsmodels.tsa.stattools import grangercausalitytests
# H₀: Aviation stress does NOT Granger-cause market volatility
# Reject H₀ if p < 0.05 → Signal has predictive power 
```

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Add your FlightAware key to unlock live departures:
```env
FLIGHTAWARE_API_KEY=your_key_here
```

### 3. Launch the Dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## 📸 Screenshots

<div align="center">
  <img width="100%" alt="Research Dashboard" src="docs/images/dashboard_overview.png" />
  <br/>
  <img width="100%" alt="Route Analysis" src="docs/images/dashboard_overlap.png" />
</div>

---

**Developed for Portfolio Showcase** — Highlighting end-to-end quantitative engineering, from raw alternative data ingestion to interactive signal validation and econometric proof.
