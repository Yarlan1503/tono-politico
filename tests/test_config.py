"""Tests para config tipada del pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from tono_politico.config import Config, load_config


def test_load_config_minimo_aplica_defaults(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    cfg = load_config(config_path)

    assert isinstance(cfg, Config)
    assert cfg.project.data_dir == Path("data")
    assert cfg.project.output_dir == Path("output")
    assert cfg.project.idioma == "es"
    assert cfg.ingesta.whisper_model == "large-v3-turbo"
    assert cfg.ingesta.idioma == "es"
    assert cfg.diarizacion.actor_objetivo == "Lilly Téllez"
    assert cfg.diarizacion.video_ref_id == "su9nURIj9XQ"
    assert cfg.diarizacion.pipeline == "pyannote/speaker-diarization-community-1"
    assert cfg.diarizacion.fallback_pipeline == "pyannote-community/speaker-diarization-community-1"
    assert cfg.diarizacion.device == "auto"
    assert cfg.diarizacion.umbral_match == 0.5
    assert cfg.diarizacion.umbral_ambiguo == 0.7


def test_ingesta_data_dir_debe_coincidir_con_project_data_dir(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "project:\n  data_dir: data\ningesta:\n  data_dir: otro-data\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ingesta.data_dir"):
        load_config(config_path)


def test_ingesta_data_dir_compatible_se_normaliza_a_project(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "project:\n  data_dir: data\ningesta:\n  data_dir: data\n  whisper_model: tiny\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.project.data_dir == Path("data")
    assert cfg.ingesta.whisper_model == "tiny"


def test_config_legacy_dict_no_contiene_tokens_ni_secretos(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "project:\n  data_dir: data\ndiarizacion:\n  actor_objetivo: Lilly Téllez\n",
        encoding="utf-8",
    )

    legacy = load_config(config_path).as_legacy_dict()
    rendered = repr(legacy).lower()

    assert legacy["project"]["data_dir"] == "data"
    assert legacy["ingesta"]["data_dir"] == "data"
    assert legacy["diarizacion"]["fallback_pipeline"] == (
        "pyannote-community/speaker-diarization-community-1"
    )
    assert legacy["diarizacion"]["device"] == "auto"
    assert legacy["diarizacion"]["actor_objetivo"] == "Lilly Téllez"
    assert "token" not in rendered
    assert "secret" not in rendered
