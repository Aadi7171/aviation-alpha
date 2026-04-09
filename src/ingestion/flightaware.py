"""
FlightAware AeroAPI v4 Client
================================
Live flight data ingestion via FlightAware AeroAPI.

API Docs: https://www.flightaware.com/aeroapi/portal/documentation
Free tier: 500 requests/month (Starter plan)

Set FLIGHTAWARE_API_KEY in your .env file.
If the key is absent, the client returns realistic simulated live data
so the entire pipeline runs fully offline.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.utils import CONFIG, ROOT, ensure_dirs, get_logger

logger = get_logger(__name__)

API_KEY  = os.getenv("FLIGHTAWARE_API_KEY", "")
BASE_URL = CONFIG["flightaware"]["base_url"]
CACHE_DIR = ROOT / CONFIG["data"]["raw_dir"] / "flightaware_cache"
CACHE_TTL = CONFIG["flightaware"]["cache_ttl_minutes"] * 60  # seconds


# ── HTTP Client ───────────────────────────────────────────────────────────────

class FlightAwareClient:
    """
    Thin wrapper around FlightAware AeroAPI v4.

    All methods fall back to synthetic data if API_KEY is not set.
    """

    def __init__(self, api_key: str = API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"x-apikey": self.api_key})
        ensure_dirs(CONFIG)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(" ", "_")
        return CACHE_DIR / f"{safe}.json"

    def _load_cache(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > CACHE_TTL:
            logger.debug("Cache stale for %s", key)
            return None
        with open(p) as f:
            return json.load(f)

    def _save_cache(self, key: str, data: dict) -> None:
        with open(self._cache_path(key), "w") as f:
            json.dump(data, f)

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """Make authenticated GET request with cache and fallback."""
        cache_key = endpoint + str(sorted((params or {}).items()))
        cached = self._load_cache(cache_key)
        if cached:
            logger.debug("Cache hit: %s", endpoint)
            return cached

        if not self.api_key or self.api_key == "your_api_key_here":
            logger.warning(
                "No FlightAware API key — returning simulated data for: %s", endpoint
            )
            return self._simulate_response(endpoint)

        url = BASE_URL + endpoint
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            self._save_cache(cache_key, data)
            return data
        except requests.exceptions.RequestException as exc:
            logger.error("FlightAware API error: %s — using simulated data", exc)
            return self._simulate_response(endpoint)

    # ── Simulation ────────────────────────────────────────────────────────────

    def _simulate_response(self, endpoint: str) -> dict:
        """Return realistic simulated AeroAPI response."""
        rng = np.random.default_rng(abs(hash(endpoint)) % (2**32))
        carriers = CONFIG["airlines"]["bts_codes"]
        airports = CONFIG["flightaware"]["major_airports"]

        now = datetime.now(timezone.utc)
        flights = []
        for i in range(rng.integers(8, 25)):
            carrier = rng.choice(carriers)
            dep_offset = int(rng.integers(-120, 240))
            scheduled = now + timedelta(minutes=dep_offset)
            delay = int(rng.integers(0, 90)) if rng.random() < 0.3 else 0
            cancelled = rng.random() < 0.04

            flights.append({
                "ident": f"{carrier}{rng.integers(100, 9999)}",
                "operator": carrier,
                "origin": {"code_iata": rng.choice([a.lstrip("K") for a in airports])},
                "destination": {"code_iata": rng.choice([a.lstrip("K") for a in airports])},
                "scheduled_out": scheduled.isoformat(),
                "actual_out": None if cancelled else (scheduled + timedelta(minutes=delay)).isoformat(),
                "departure_delay": delay if not cancelled else None,
                "cancelled": cancelled,
                "status": "Cancelled" if cancelled else ("Delayed" if delay > 15 else "On Time"),
                "aircraft_type": rng.choice(["B737", "A320", "B787", "A321", "E175"]),
            })
        return {"flights": flights, "num_pages": 1}

    # ── Public API Methods ────────────────────────────────────────────────────

    def get_airport_departures(
        self,
        icao: str,
        hours_back: int = 2,
        max_pages: int = 1,
    ) -> pd.DataFrame:
        """
        Fetch recent departures from an airport.

        Args:
            icao:       ICAO airport code (e.g. 'KATL').
            hours_back: Look back N hours from now.
            max_pages:  Maximum result pages to fetch.

        Returns:
            DataFrame of flight records.
        """
        endpoint = f"/airports/{icao}/flights/departures"
        data = self._get(endpoint, params={"max_pages": max_pages})
        flights = data.get("flights", [])

        records = []
        for f in flights:
            records.append({
                "ident":           f.get("ident"),
                "carrier":         f.get("operator"),
                "origin":          (f.get("origin") or {}).get("code_iata"),
                "destination":     (f.get("destination") or {}).get("code_iata"),
                "scheduled_out":   f.get("scheduled_out"),
                "actual_out":      f.get("actual_out"),
                "departure_delay": f.get("departure_delay", 0) or 0,
                "cancelled":       bool(f.get("cancelled", False)),
                "status":          f.get("status", "Unknown"),
                "aircraft_type":   f.get("aircraft_type"),
                "airport":         icao,
                "fetched_at":      datetime.now(timezone.utc).isoformat(),
            })

        return pd.DataFrame(records) if records else pd.DataFrame()

    def get_all_airports_snapshot(self) -> pd.DataFrame:
        """
        Fetch departure snapshots from all configured major airports.

        Returns:
            Combined DataFrame of flights across all major airports.
        """
        airports = CONFIG["flightaware"]["major_airports"]
        frames = []
        for icao in airports:
            logger.info("Fetching departures: %s", icao)
            df = self.get_airport_departures(icao)
            if not df.empty:
                frames.append(df)
            time.sleep(0.3)  # Be polite to the API

        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()

    def compute_live_disruption_scores(self) -> pd.DataFrame:
        """
        Compute a real-time disruption score per carrier from live snapshot.

        Returns:
            DataFrame with [carrier, cancel_rate, avg_delay, disruption_score].
        """
        snap = self.get_all_airports_snapshot()
        if snap.empty:
            logger.warning("Empty live snapshot — returning empty scores")
            return pd.DataFrame()

        snap["departure_delay"] = pd.to_numeric(snap["departure_delay"], errors="coerce").fillna(0)
        snap["cancelled"]       = snap["cancelled"].astype(bool)

        grp = snap.groupby("carrier")
        scores = grp.agg(
            total_flights  = ("ident", "count"),
            cancel_rate    = ("cancelled", "mean"),
            avg_delay      = ("departure_delay", "mean"),
        ).reset_index()

        # Simple composite disruption score [0,1]
        max_delay = scores["avg_delay"].max()
        delay_norm = scores["avg_delay"] / (max_delay if max_delay > 1 else 1.0)
        scores["disruption_score"] = (
            0.5 * scores["cancel_rate"] + 0.5 * delay_norm
        ).clip(0, 1)

        scores["timestamp"] = datetime.now(timezone.utc).isoformat()
        return scores.sort_values("disruption_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    client = FlightAwareClient()
    print("=== Live Disruption Scores (or simulated) ===")
    scores = client.compute_live_disruption_scores()
    print(scores.to_string())
    print("\n=== KATL Departures Sample ===")
    deps = client.get_airport_departures("KATL")
    print(deps.head(10).to_string())
