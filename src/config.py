"""Configuration loader for the trading system."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_data_dir(config: Dict[str, Any]) -> Path:
    """Retrieve the raw data directory path from configuration.
    
    Args:
        config: Loaded configuration dictionary.
        
    Returns:
        Path object pointing to the raw data directory.
    """
    return Path(config.get("data_dir", "data/raw"))
