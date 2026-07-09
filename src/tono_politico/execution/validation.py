"""Validación cruzada de ``RunConfig``."""

from __future__ import annotations

from pathlib import Path

from .models import RunConfig, StageName


class ConfigValidationError(ValueError):
    """Error de configuración dirigido al usuario final."""


def validate_run_config(cfg: RunConfig) -> None:
    """Valida dependencias entre stages y parámetros de ejecución."""
    _validate_run_settings(cfg)
    _validate_stage_enabled(cfg)
    _validate_stage_dependencies(cfg)
    _validate_thresholds(cfg)
    _validate_argument_shape(cfg)
    _validate_topics_cluster(cfg)


def _validate_run_settings(cfg: RunConfig) -> None:
    if cfg.run.max_videos is not None and cfg.run.max_videos <= 0:
        raise ConfigValidationError("run.max_videos debe ser null o entero positivo")


def _validate_stage_enabled(cfg: RunConfig) -> None:
    for stage in cfg.run.stages:
        if stage == "speech2text" and not cfg.speech2text.enabled:
            raise ConfigValidationError(
                "speech2text.enabled=false pero stage speech2text está activo"
            )
        if stage == "argument_shape" and not cfg.discursive_approach.argument_shape.enabled:
            raise ConfigValidationError(
                "discursive_approach.argument_shape.enabled=false pero stage "
                "argument_shape está activo"
            )
        if stage == "topics_cluster" and not cfg.discursive_approach.topics_cluster.enabled:
            raise ConfigValidationError(
                "discursive_approach.topics_cluster.enabled=false pero stage "
                "topics_cluster está activo"
            )
        if stage == "topics_approach" and not cfg.discursive_approach.topics_approach.enabled:
            raise ConfigValidationError(
                "discursive_approach.topics_approach.enabled=false pero stage "
                "topics_approach está activo"
            )


def _validate_stage_dependencies(cfg: RunConfig) -> None:
    produced: set[StageName] = set()
    for stage in cfg.run.stages:
        if stage == "speech2text":
            if not cfg.input.playlist_url:
                raise ConfigValidationError("input.playlist_url es requerido por stage speech2text")
            produced.add(stage)
            continue

        if stage == "argument_shape":
            if "speech2text" not in produced and _transcripts_dir(cfg) is None:
                raise ConfigValidationError(
                    "argument_shape requiere speech2text previo o "
                    "input.actor_transcripts_dir existente"
                )
            produced.add(stage)
            continue

        if stage == "topics_cluster":
            if "argument_shape" not in produced and not _existing_file(cfg.input.argumentos_path):
                raise ConfigValidationError(
                    "topics_cluster requiere argument_shape previo o "
                    "input.argumentos_path existente"
                )
            produced.add(stage)
            continue

        if stage == "topics_approach":
            if "topics_cluster" not in produced and not _existing_file(cfg.input.temas_path):
                raise ConfigValidationError(
                    "topics_approach requiere topics_cluster previo o input.temas_path existente"
                )
            produced.add(stage)


def _transcripts_dir(cfg: RunConfig) -> Path | None:
    candidates = [
        cfg.input.actor_transcripts_dir,
        cfg.discursive_approach.input.actor_transcripts_dir,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _existing_file(path: Path | None) -> bool:
    return path is not None and path.exists() and path.is_file()


def _validate_thresholds(cfg: RunConfig) -> None:
    speaker = cfg.speech2text.speaker_timestamps
    if speaker.umbral_match >= speaker.umbral_ambiguo:
        raise ConfigValidationError(
            "speech2text.speaker_timestamps.umbral_match debe ser menor que umbral_ambiguo"
        )


def _validate_argument_shape(cfg: RunConfig) -> None:
    shape = cfg.discursive_approach.argument_shape
    if not 0 <= shape.breakpoint_percentile <= 100:
        raise ConfigValidationError(
            "discursive_approach.argument_shape.breakpoint_percentile debe estar entre 0 y 100"
        )
    if shape.min_oraciones < 1:
        raise ConfigValidationError(
            "discursive_approach.argument_shape.min_oraciones debe ser >= 1"
        )
    if shape.max_oraciones < shape.min_oraciones:
        raise ConfigValidationError(
            "discursive_approach.argument_shape.max_oraciones debe ser >= min_oraciones"
        )
    if shape.max_palabras < 1:
        raise ConfigValidationError("discursive_approach.argument_shape.max_palabras debe ser >= 1")


def _validate_topics_cluster(cfg: RunConfig) -> None:
    cluster = cfg.discursive_approach.topics_cluster
    if cluster.min_topic_size < 1:
        raise ConfigValidationError(
            "discursive_approach.topics_cluster.min_topic_size debe ser >= 1"
        )
    if cluster.n_neighbors < 2:
        raise ConfigValidationError("discursive_approach.topics_cluster.n_neighbors debe ser >= 2")
    if cluster.n_components < 2:
        raise ConfigValidationError("discursive_approach.topics_cluster.n_components debe ser >= 2")
