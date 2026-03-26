"""
This utility file provides functions to load configuration files, mainly used for defining settings and parameters for various algorithms.

Authors: Arnaud Poletto
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from types import SimpleNamespace


def load_config(file_path: str) -> Dict:
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        file_path (str): The path to the YAML configuration file.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.

    Returns:
        config (Dict): The contents of the configuration file as a dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"❌ Configuration file not found: {Path.resolve(Path(file_path))}"
        )

    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    if config is None:
        print("⚠️  Configuration file is empty, using empty configuration.")
        config = {}

    return config


def dict_to_namespace(d: Any) -> Any:
    """
    Recursively convert a dictionary to a SimpleNamespace for attribute-style access.

    Args:
        d: The dictionary to convert, or the non-dict value to return as-is.

    Returns:
        namespace (Any): The converted namespace or original value.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d