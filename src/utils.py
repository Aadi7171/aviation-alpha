"""
Aviation Alpha Research Pipeline
=================================
Shared utilities: config loading, logging, path helpers.
"""

import os
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

# ── Project Root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    """Load config.yaml from project root."""
    config_path = ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a coloured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s › %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dirs(cfg: dict) -> None:
    """Create raw / processed data directories if they don't exist."""
    (ROOT / cfg["data"]["raw_dir"]).mkdir(parents=True, exist_ok=True)
    (ROOT / cfg["data"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
    (ROOT / cfg["data"]["raw_dir"] / "flightaware_cache").mkdir(
        parents=True, exist_ok=True
    )


CONFIG = load_config()
