"""Tests de validación cruzada del config de ejecución."""

from __future__ import annotations

from pathlib import Path

import pytest

from tono_politico.execution.config import load_run_config
from tono_politico.execution.validation import ConfigValidationError, validate_run_config


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "run-config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def _load(tmp_path: Path, content: str):
    return load_run_config(_write_config(tmp_path, content))


def test_validate_speech2text_requiere_playlist_url(tmp_path: Path):
    cfg = _load(
        tmp_path,
        """
schema_version: tono-politico.run.v1
run:
  stages: [speech2text]
""",
    )

    with pytest.raises(ConfigValidationError, match="input.playlist_url"):
        validate_run_config(cfg)


def test_validate_argument_shape_requiere_transcripts_si_no_hay_stage_previo(tmp_path: Path):
    cfg = _load(
        tmp_path,
        """
schema_version: tono-politico.run.v1
run:
  stages: [argument_shape]
""",
    )

    with pytest.raises(ConfigValidationError, match="actor_transcripts_dir"):
        validate_run_config(cfg)


def test_validate_argument_shape_acepta_transcripts_dir_existente(tmp_path: Path):
    transcripts_dir = tmp_path / "actor_transcripts"
    transcripts_dir.mkdir()
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  stages: [argument_shape]
input:
  actor_transcripts_dir: {transcripts_dir}
""",
    )

    validate_run_config(cfg)


def test_validate_topics_cluster_requiere_argumentos_si_no_hay_stage_previo(tmp_path: Path):
    cfg = _load(
        tmp_path,
        """
schema_version: tono-politico.run.v1
run:
  stages: [topics_cluster]
""",
    )

    with pytest.raises(ConfigValidationError, match="input.argumentos_path"):
        validate_run_config(cfg)


def test_validate_topics_approach_requiere_temas_si_no_hay_stage_previo(tmp_path: Path):
    cfg = _load(
        tmp_path,
        """
schema_version: tono-politico.run.v1
run:
  stages: [topics_approach]
""",
    )

    with pytest.raises(ConfigValidationError, match="input.temas_path"):
        validate_run_config(cfg)


def test_validate_rechaza_stage_deshabilitado(tmp_path: Path):
    cfg = _load(
        tmp_path,
        """
schema_version: tono-politico.run.v1
run:
  stages: [speech2text]
input:
  playlist_url: url
speech2text:
  enabled: false
""",
    )

    with pytest.raises(ConfigValidationError, match="speech2text.enabled"):
        validate_run_config(cfg)


def test_validate_rechaza_thresholds_invalidos(tmp_path: Path):
    cfg = _load(
        tmp_path,
        """
schema_version: tono-politico.run.v1
run:
  stages: [speech2text]
input:
  playlist_url: url
speech2text:
  speaker_timestamps:
    umbral_match: 0.8
    umbral_ambiguo: 0.7
""",
    )

    with pytest.raises(ConfigValidationError, match="umbral_match"):
        validate_run_config(cfg)


def test_validate_rechaza_parametros_argument_shape_invalidos(tmp_path: Path):
    cfg = _load(
        tmp_path,
        """
schema_version: tono-politico.run.v1
run:
  stages: [speech2text, argument_shape]
input:
  playlist_url: url
discursive_approach:
  argument_shape:
    breakpoint_percentile: 101
""",
    )

    with pytest.raises(ConfigValidationError, match="breakpoint_percentile"):
        validate_run_config(cfg)
