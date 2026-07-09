"""Carga YAML del config de ejecución stage-based."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .models import RunConfig


def load_run_config(path: Path) -> RunConfig:
    """Carga un YAML ``tono-politico.run.v1`` y devuelve ``RunConfig`` tipado."""
    if not path.exists():
        raise FileNotFoundError(f"Config no encontrado: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config debe ser un mapping YAML: {path}")
    return RunConfig.from_mapping(raw)


def is_run_config_mapping(data: Mapping[str, Any]) -> bool:
    """Devuelve True si el mapping declara el schema nuevo de ejecución."""
    return data.get("schema_version") == "tono-politico.run.v1"


def is_run_config_file(path: Path) -> bool:
    """Inspección liviana para que main.py distinga config nuevo vs legacy."""
    if not path.exists():
        return False
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return isinstance(raw, Mapping) and is_run_config_mapping(raw)
