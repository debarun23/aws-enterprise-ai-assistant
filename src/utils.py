"""
src/utils.py — Shared utilities: config loading, logging setup.
"""

import logging
import yaml
from pathlib import Path
from typing import Any


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Load and return the YAML config as a nested dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found at: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str, config: dict[str, Any] | None = None) -> logging.Logger:
    """Return a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_str = "INFO"
    if config and "logging" in config:
        level_str = config["logging"].get("level", "INFO")

    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if config and "logging" in config:
        log_dir = Path(config["logging"].get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger