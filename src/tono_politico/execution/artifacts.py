"""Rutas y serialización liviana de artefactos de ejecución."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tono_politico.discursive_approach.argument_shape.models import Argumento, Oracion

from .models import ArtifactKey, ArtifactPaths, RunConfig


def resolve_artifacts(cfg: RunConfig, run_id: str) -> ArtifactPaths:
    """Resuelve las rutas durables de una corrida."""
    run_dir = cfg.output.run_dir if cfg.output.run_dir is not None else cfg.output.base_dir / run_id
    return ArtifactPaths(
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        resolved_config_path=run_dir / "resolved-config.yaml",
        actor_transcripts_dir=run_dir / "speech2text" / "actor_transcripts",
        argumentos_path=run_dir / "discursive" / "argumentos.json",
        temas_path=run_dir / "discursive" / "discursive-temas.json",
        enfoques_path=run_dir / "discursive" / "discursive-enfoques.json",
    )


def artifact_exists(paths: ArtifactPaths, key: ArtifactKey) -> bool:
    """Devuelve si el artefacto de salida asociado a ``key`` existe."""
    path = _path_for_key(paths, key)
    if key == "actor_transcripts_dir":
        return path.exists() and path.is_dir()
    if key == "playlist_url":
        return False
    return path.exists() and path.is_file()


def guardar_argumentos(argumentos: list[Argumento], path: Path | str) -> Path:
    """Persiste ``Argumento[]`` como artefacto JSON simple."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": "discursive_argumentos.v1",
        "argumentos": [_argumento_to_dict(argumento) for argumento in argumentos],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def cargar_argumentos(path: Path | str) -> list[Argumento]:
    """Carga ``Argumento[]`` desde el artefacto JSON simple."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [_argumento_from_dict(item) for item in data.get("argumentos", [])]


def _path_for_key(paths: ArtifactPaths, key: ArtifactKey) -> Path:
    if key == "actor_transcripts_dir":
        return paths.actor_transcripts_dir
    if key == "argumentos_path":
        return paths.argumentos_path
    if key == "temas_path":
        return paths.temas_path
    if key == "enfoques_path":
        return paths.enfoques_path
    return paths.run_dir


def _oracion_to_dict(oracion: Oracion) -> dict[str, Any]:
    return {"texto": oracion.texto, "t_start": oracion.t_start, "t_end": oracion.t_end}


def _oracion_from_dict(data: dict[str, Any]) -> Oracion:
    return Oracion(texto=data["texto"], t_start=data["t_start"], t_end=data["t_end"])


def _argumento_to_dict(argumento: Argumento) -> dict[str, Any]:
    return {
        "texto": argumento.texto,
        "t_start": argumento.t_start,
        "t_end": argumento.t_end,
        "oraciones": [_oracion_to_dict(oracion) for oracion in argumento.oraciones],
        "word_count": argumento.word_count,
        "video_id": argumento.video_id,
        "fecha": argumento.fecha,
    }


def _argumento_from_dict(data: dict[str, Any]) -> Argumento:
    return Argumento(
        texto=data["texto"],
        t_start=data["t_start"],
        t_end=data["t_end"],
        oraciones=[_oracion_from_dict(item) for item in data.get("oraciones", [])],
        word_count=data.get("word_count", 0),
        video_id=data.get("video_id", ""),
        fecha=data.get("fecha"),
    )
