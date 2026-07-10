"""Rutas y serialización liviana de artefactos de ejecución."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tono_politico.discursive_approach.argument_shape.models import Argumento, Oracion
from tono_politico.speech2text.models import ActorTranscript, ActorTranscriptSegment, AsrMetadata

from .models import ArtifactKey, ArtifactPaths, RunConfig


def resolve_artifacts(cfg: RunConfig, run_id: str) -> ArtifactPaths:
    """Resuelve las rutas durables de una corrida."""
    run_dir = cfg.output.run_dir if cfg.output.run_dir is not None else cfg.output.base_dir / run_id
    return ArtifactPaths(
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        resolved_config_path=run_dir / "resolved-config.yaml",
        actor_transcripts_dir=run_dir / "speech2text" / "actor_transcripts",
        speech2text_quality_path=run_dir / "speech2text" / "quality.json",
        argumentos_path=run_dir / "discursive" / "argumentos.json",
        temas_path=run_dir / "discursive" / "discursive-temas.json",
        enfoques_path=run_dir / "discursive" / "discursive-enfoques.json",
    )


def artifact_exists(paths: ArtifactPaths, key: ArtifactKey) -> bool:
    """Devuelve si el artefacto de salida asociado a ``key`` existe."""
    path = _path_for_key(paths, key)
    if key == "actor_transcripts_dir":
        return (
            path.exists()
            and path.is_dir()
            and any(
                candidate.is_file()
                and candidate.suffix == ".json"
                and candidate.stat().st_size > 0
                and _valid_actor_transcript_file(candidate)
                for candidate in path.iterdir()
            )
        )
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


def _segment_to_dict(segment: ActorTranscriptSegment) -> dict[str, Any]:
    return {
        "text": segment.text,
        "t_start": segment.t_start,
        "t_end": segment.t_end,
        "speaker": segment.speaker,
        "source_turn": {
            "t_start": segment.source_turn_start,
            "t_end": segment.source_turn_end,
        },
        "word_count": segment.word_count,
    }


def _segment_from_dict(data: dict[str, Any]) -> ActorTranscriptSegment:
    source_turn = data.get("source_turn")
    if isinstance(source_turn, dict):
        source_start = source_turn["t_start"]
        source_end = source_turn["t_end"]
    else:
        source_start = data.get("source_turn_start", data["t_start"])
        source_end = data.get("source_turn_end", data["t_end"])
    return ActorTranscriptSegment(
        text=data["text"],
        t_start=data["t_start"],
        t_end=data["t_end"],
        speaker=data["speaker"],
        source_turn_start=source_start,
        source_turn_end=source_end,
        word_count=data["word_count"],
    )


def actor_transcript_to_dict(transcript: ActorTranscript) -> dict[str, Any]:
    """Serializa ``ActorTranscript`` al contrato JSON actor_transcript.v1."""
    data: dict[str, Any] = {
        "schema_version": transcript.schema_version,
        "video_id": transcript.video_id,
        "actor": transcript.actor,
        "scope": transcript.scope,
        "asr": {
            "provider": transcript.asr.provider,
            "model": transcript.asr.model,
            "language": transcript.asr.language,
        },
        "segments": [_segment_to_dict(segment) for segment in transcript.segments],
    }
    if transcript.fecha is not None:
        data["fecha"] = transcript.fecha
    return data


def actor_transcript_to_json(transcript: ActorTranscript) -> str:
    return json.dumps(
        actor_transcript_to_dict(transcript),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _valid_actor_transcript_file(path: Path) -> bool:
    try:
        cargar_actor_transcript(path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return True


def actor_transcript_from_json(json_str: str) -> ActorTranscript:
    data = json.loads(json_str)
    if not isinstance(data, dict):
        raise ValueError("ActorTranscript debe ser un objeto JSON")
    _validar_actor_transcript_data(data)
    asr = data["asr"]
    return ActorTranscript(
        schema_version=data["schema_version"],
        video_id=data["video_id"],
        actor=data["actor"],
        scope=data["scope"],
        asr=AsrMetadata(
            provider=asr["provider"],
            model=asr["model"],
            language=asr["language"],
        ),
        segments=[_segment_from_dict(segment) for segment in data.get("segments", [])],
        fecha=data.get("fecha"),
    )


def _validar_actor_transcript_data(data: dict[str, Any]) -> None:
    if data.get("schema_version") != "actor_transcript.v1":
        raise ValueError("schema_version incompatible con actor_transcript.v1")
    if not data.get("video_id") or not data.get("actor"):
        raise ValueError("video_id y actor son obligatorios")
    if data.get("scope") != "actor_only":
        raise ValueError("scope debe ser actor_only")
    asr = data.get("asr")
    if not isinstance(asr, dict) or not all(
        asr.get(key) for key in ("provider", "model", "language")
    ):
        raise ValueError("asr debe contener provider, model y language")
    segments = data.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("segments debe ser una lista")
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("cada segmento debe ser un objeto")
        try:
            t_start = float(segment["t_start"])
            t_end = float(segment["t_end"])
            source_turn = segment.get("source_turn")
            source_start = float(
                source_turn["t_start"]
                if isinstance(source_turn, dict)
                else segment.get("source_turn_start", segment["t_start"])
            )
            source_end = float(
                source_turn["t_end"]
                if isinstance(source_turn, dict)
                else segment.get("source_turn_end", segment["t_end"])
            )
            word_count = int(segment["word_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("segmento actor_transcript incompleto") from exc
        if t_end <= t_start or source_end <= source_start:
            raise ValueError("t_end debe ser mayor que t_start")
        if word_count < 0:
            raise ValueError("word_count no puede ser negativo")


def guardar_actor_transcript(transcript: ActorTranscript, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(actor_transcript_to_json(transcript), encoding="utf-8")
    return path


def cargar_actor_transcript(path: Path | str) -> ActorTranscript:
    path = Path(path)
    return actor_transcript_from_json(path.read_text(encoding="utf-8"))
