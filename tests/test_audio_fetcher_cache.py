"""Tests de rutas de cache de audio_fetcher."""

from __future__ import annotations

from pathlib import Path

from tono_politico.speech2text.audio_fetcher.audio import (
    DATA_DIR,
    ruta_audio,
    ruta_dir_videos,
)


def test_data_dir_default() -> None:
    assert DATA_DIR == Path("data")


def test_ruta_dir_videos_default() -> None:
    assert ruta_dir_videos("MiPlay") == Path("data/MiPlay/videos-MiPlay")


def test_ruta_dir_videos_base_dir(tmp_path: Path) -> None:
    assert ruta_dir_videos("P", tmp_path) == tmp_path / "P" / "videos-P"


def test_ruta_audio(tmp_path: Path) -> None:
    assert ruta_audio("P", "vid001", tmp_path) == tmp_path / "P" / "videos-P" / "vid001.wav"


def test_sin_rutas_de_transcripcion() -> None:
    """audio_fetcher no debe exponer rutas de transcripciones JSON."""
    import tono_politico.speech2text.audio_fetcher.audio as audio_mod

    assert not hasattr(audio_mod, "ruta_transcripcion")
    assert not hasattr(audio_mod, "ruta_dir_transcripciones")
