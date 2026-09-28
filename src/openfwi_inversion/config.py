"""Configuration loading, path resolution, and validation."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ConfigError("The YAML root must be a mapping.")

    mode = config.get("mode")
    if mode not in {"diffusion", "optimization", "train"}:
        raise ConfigError("mode must be diffusion, optimization, or train.")

    selected = config.get("dataset")
    datasets = config.get("datasets", {})
    if selected not in datasets:
        raise ConfigError(f"Unknown dataset {selected!r}; define it under datasets.")

    config["_config_dir"] = config_path.parent
    config["_config_path"] = config_path
    config["_dataset"] = datasets[selected]
    return config


def resolve_path(config: dict[str, Any], value: str) -> Path:
    """Resolve a user path relative to the YAML file rather than the shell cwd."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config["_config_dir"] / path).resolve()


def require(mapping: dict[str, Any], key: str, section: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ConfigError(f"Missing required setting: {section}.{key}") from exc


def snapshot_config(config: dict[str, Any], output_dir: Path) -> None:
    """Save the exact input YAML beside the generated results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config["_config_path"], output_dir / "config.yaml")
