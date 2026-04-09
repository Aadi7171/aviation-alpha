"""
OpenFlights Route & Airport Data Loader
=========================================
Parses static OpenFlights datasets (routes.dat, airports.dat).
These are free, no API key required.

Data source:
  https://openflights.org/data.html
  routes.dat  — ~67k routes
  airports.dat — ~7k airports

Falls back to synthetic data if local files are not present.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.utils import CONFIG, ROOT, ensure_dirs, get_logger

logger = get_logger(__name__)

# ── OpenFlights URLs ──────────────────────────────────────────────────────────
ROUTES_URL   = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"
AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"

ROUTES_COLS = [
    "airline", "airline_id", "src_airport", "src_airport_id",
    "dst_airport", "dst_airport_id", "codeshare", "stops", "equipment",
]
AIRPORT_COLS = [
    "airport_id", "name", "city", "country", "iata", "icao",
    "lat", "lon", "alt", "timezone", "dst", "tz_db", "type", "source",
]


def _download_openflights(url: str, cols: list[str]) -> pd.DataFrame | None:
    """Try to download an OpenFlights CSV. Returns None on failure."""
    try:
        logger.info("Fetching: %s", url)
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(
            io.StringIO(resp.text),
            header=None,
            names=cols,
            na_values=["\\N"],
        )
        return df
    except Exception as exc:
        logger.warning("Could not download OpenFlights data (%s): %s", url, exc)
        return None


def _synthetic_routes(carriers: list[str], airports: list[str]) -> pd.DataFrame:
    """Generate plausible synthetic route data when offline."""
    logger.info("Generating synthetic OpenFlights route data…")
    rng = np.random.default_rng(99)
    records = []
    for carrier in carriers:
        n = rng.integers(40, 120)
        srcs = rng.choice(airports, size=n)
        dsts = rng.choice(airports, size=n)
        for s, d in zip(srcs, dsts):
            if s != d:
                records.append({"airline": carrier, "src_airport": s, "dst_airport": d})
    return pd.DataFrame(records).drop_duplicates()


def load_routes(use_live: bool = True) -> pd.DataFrame:
    """
    Load airline route pairs.

    Args:
        use_live: Attempt to download from OpenFlights before falling back to synthetic.

    Returns:
        DataFrame with columns [airline, src_airport, dst_airport].
    """
    ensure_dirs(CONFIG)
    cache = ROOT / CONFIG["data"]["raw_dir"] / "openflights_routes.parquet"

    if cache.exists():
        logger.info("Loading routes from cache: %s", cache)
        return pd.read_parquet(cache)

    df = None
    if use_live:
        df = _download_openflights(ROUTES_URL, ROUTES_COLS)
        if df is not None:
            df = df[["airline", "src_airport", "dst_airport"]].dropna()
            # Filter to our carriers of interest (and keep rest for graph completeness)
            logger.info("OpenFlights routes loaded: %d rows", len(df))

    if df is None:
        airports = [
            "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "LAS",
            "MIA", "IAH", "SEA", "CLT", "PHX", "MCO", "EWR", "MSP",
            "BOS", "DTW", "PHL", "LGA", "FLL", "BWI", "SLC", "IAD",
        ]
        df = _synthetic_routes(CONFIG["airlines"]["bts_codes"], airports)

    df.to_parquet(cache, index=False)
    return df


def load_airports(use_live: bool = True) -> pd.DataFrame:
    """
    Load airport metadata (name, city, country, lat/lon).

    Returns:
        DataFrame with columns [iata, name, city, country, lat, lon].
    """
    ensure_dirs(CONFIG)
    cache = ROOT / CONFIG["data"]["raw_dir"] / "openflights_airports.parquet"

    if cache.exists():
        logger.info("Loading airports from cache: %s", cache)
        return pd.read_parquet(cache)

    df = None
    if use_live:
        df = _download_openflights(AIRPORTS_URL, AIRPORT_COLS)
        if df is not None:
            df = df[["iata", "name", "city", "country", "lat", "lon"]].dropna(subset=["iata"])
            df = df[df["iata"] != "\\N"]

    if df is None:
        # Minimal synthetic airport table
        airports = {
            "ATL": ("Hartsfield-Jackson", "Atlanta", "USA", 33.64, -84.43),
            "LAX": ("Los Angeles Intl", "Los Angeles", "USA", 33.94, -118.41),
            "ORD": ("O'Hare International", "Chicago", "USA", 41.98, -87.90),
            "DFW": ("Dallas/Fort Worth", "Dallas", "USA", 32.90, -97.04),
            "DEN": ("Denver International", "Denver", "USA", 39.86, -104.67),
            "JFK": ("John F. Kennedy", "New York", "USA", 40.64, -73.78),
            "SFO": ("San Francisco Intl", "San Francisco", "USA", 37.62, -122.38),
            "MIA": ("Miami International", "Miami", "USA", 25.80, -80.29),
        }
        rows = [{"iata": k, "name": v[0], "city": v[1], "country": v[2], "lat": v[3], "lon": v[4]}
                for k, v in airports.items()]
        df = pd.DataFrame(rows)

    df.to_parquet(cache, index=False)
    return df


def carrier_route_sets(routes_df: pd.DataFrame) -> dict[str, set]:
    """
    Build a dict mapping each carrier → set of (src, dst) route pairs.

    Args:
        routes_df: Output of load_routes().

    Returns:
        {'UA': {('JFK','LAX'), ('ORD','SFO'), …}, …}
    """
    carrier_routes: dict[str, set] = {}
    for _, row in routes_df.iterrows():
        c = row["airline"]
        pair = (str(row["src_airport"]), str(row["dst_airport"]))
        carrier_routes.setdefault(c, set()).add(pair)
    return carrier_routes


if __name__ == "__main__":
    routes = load_routes()
    airports = load_airports()
    crs = carrier_route_sets(routes)
    print(f"Routes: {len(routes)} rows")
    print(f"Airports: {len(airports)} rows")
    print(f"Carriers in route data: {len(crs)}")
    for c in list(crs)[:5]:
        print(f"  {c}: {len(crs[c])} routes")
