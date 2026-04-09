<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2942,100:185FA5&height=200&section=header&text=Aviation%20Alpha&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Alternative%20Data%20%7C%20Quant%20Research%20Pipeline&descAlignY=58&descSize=18" width="100%"/>

**Aviation Alpha** is a production-grade quantitative research pipeline designed to extract alpha from aviation-based alternative data. It bridges the gap between raw aerospace activity and actionable market intelligence using econometric causality and network graph theory.

![Dashboard](https://img.shields.io/badge/UI-Streamlit--Premium-FF4B4B)
![Backend](https://img.shields.io/badge/Backend-Python--Modular-3776AB)
![Econometrics](https://img.shields.io/badge/Econometrics-Statsmodels-00897B)

</div>

## ✨ Key Features

- **Route Overlap Engine**: Calculates Jaccard similarity across carrier networks to detect M&A pressure and competitive consolidation.
- **Disruption Radar**: Real-time rolling z-score spike detection for cancellations and delays, identifying operational distress before it hits the tape.
- **Granger Causality Suite**: Statistically verifies lead-lag relationships between aviation stress and market volatility using automated AIC lag selection.
- **Network Vulnerability Scoring**: Uses NetworkX to model hub concentration and clustering coefficients, scoring carriers on structural fragility.
- **Live FlightAware Feed**: Integrated with AeroAPI v4 for real-time global flight snapshots with smart response caching.

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
├── requirements.txt
└── .env.example
```

## 📊 Methodology

### Signal 1: Route Overlap (M&A Pressure)
High Jaccard similarity between two carriers signals route competition, which historically precedes acquisition activity.
```
Jaccard(A, B) = |routes_A ∩ routes_B| / |routes_A ∪ routes_B|
```

### Signal 2: Disruption Spikes
Abnormal cancellation or delay spikes may signal operational distress — a leading indicator of financial pressure.
```
z = (cancel_rate - rolling_mean) / rolling_std
spike = |z| > 2.5σ
```

### Granger Causality Validation
Statistical tests to determine if aviation stress signals have predictive power over market volatility.
| Cause | Effect |
|---|---|
| Market Disruption Index | Airline Market Volatility |
| Avg Route Overlap | Market Volatility |

## 🛠️ Tech Stack

- **Data Engine**: Python, Pandas, Numpy.
- **Network Theory**: NetworkX (for airline graph modelling).
- **Econometrics**: Statsmodels (Granger Causality, stationarity checks).
- **Visualization**: Streamlit, Plotly (Interactive financial charts).
- **Live Ingestion**: FlightAware AeroAPI v4, yfinance.

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/Aadi7171/aviation-alpha.git
cd aviation-alpha
pip install -r requirements.txt
```

### 2. Configure Environment
*Create a `.env` file for live flight data (optional):*
```env
FLIGHTAWARE_API_KEY=your_api_key_here
```

### 3. Launch Research Dashboard
```bash
streamlit run src/dashboard/app.py
```

## 📸 Screenshots

<div align="center">
  <img width="100%" alt="Dashboard Overview" src="docs/images/dashboard_overview.png" />
  <br/>
  <img width="100%" alt="Network Analysis" src="docs/images/dashboard_overlap.png" />
</div>

---

**Developed for Portfolio Showcase** — Focusing on demonstrating end-to-end quantitative engineering, from raw alternative data ingestion to interactive signal validation.
