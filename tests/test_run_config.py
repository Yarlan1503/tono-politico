"""Tests para el config de ejecución stage-based."""

from __future__ import annotations

from pathlib import Path

import pytest

from tono_politico.execution.config import load_run_config
from tono_politico.execution.models import RunConfig


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "run-config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_run_config_minimo_aplica_defaults(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        """
schema_version: tono-politico.run.v1
run:
  stages: [speech2text]
input:
  playlist_url: https://youtube.com/playlist?list=abc
""",
    )

    cfg = load_run_config(config_path)

    assert isinstance(cfg, RunConfig)
    assert cfg.schema_version == "tono-politico.run.v1"
    assert cfg.run.stages == ["speech2text"]
    assert cfg.run.resume is True
    assert cfg.run.overwrite is False
    assert cfg.run.keep_cache is False
    assert cfg.input.playlist_url == "https://youtube.com/playlist?list=abc"
    assert cfg.output.base_dir == Path("output")
    assert cfg.output.run_dir is None
    assert cfg.project.data_dir == Path("data")
    assert cfg.project.idioma == "es"
    assert cfg.speech2text.speaker_timestamps.actor_objetivo == "Lilly Téllez"
    assert cfg.speech2text.transcribe_speech.whisper_model == "large-v3-turbo"
    assert cfg.discursive_approach.argument_shape.spacy_model == "es_core_news_lg"
    assert cfg.discursive_approach.topics_cluster.min_topic_size == 3


def test_load_run_config_rechaza_schema_version_desconocido(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        """
schema_version: tono-politico.run.v999
run:
  stages: [speech2text]
input:
  playlist_url: url
""",
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_run_config(config_path)


def test_load_run_config_rechaza_stages_vacio(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        """
schema_version: tono-politico.run.v1
run:
  stages: []
""",
    )

    with pytest.raises(ValueError, match="run.stages"):
        load_run_config(config_path)


def test_load_run_config_rechaza_seccion_no_mapping(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        """
schema_version: tono-politico.run.v1
run: invalid
""",
    )

    with pytest.raises(ValueError, match="run"):
        load_run_config(config_path)


def test_load_run_config_normaliza_paths(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        """
schema_version: tono-politico.run.v1
run:
  stages: [argument_shape]
input:
  actor_transcripts_dir: output/run-1/speech2text/actor_transcripts
output:
  base_dir: custom-output
  run_dir: custom-output/run-1
project:
  data_dir: custom-data
""",
    )

    cfg = load_run_config(config_path)

    assert cfg.input.actor_transcripts_dir == Path("output/run-1/speech2text/actor_transcripts")
    assert cfg.output.base_dir == Path("custom-output")
    assert cfg.output.run_dir == Path("custom-output/run-1")
    assert cfg.project.data_dir == Path("custom-data")
