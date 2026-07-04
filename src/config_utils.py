"""Shared utility for loading the YAML config used across the whole pipeline."""

import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the project YAML config into a plain dict.

    Keeping this in one place means every script (data prep, train,
    evaluate, inference, app) reads the exact same config format.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path.resolve()}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config
