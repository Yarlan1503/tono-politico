"""Tests de DTOs de audio_fetcher (VideoMeta, AudioVideo, DownloadResult)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tono_politico.speech2text.audio_fetcher.models import (
    AudioVideo,
    DownloadResult,
    VideoMeta,
)


def test_video_meta_fields() -> None:
    meta = VideoMeta(
        video_id="abc123",
        url="https://www.youtube.com/watch?v=abc123",
        titulo="Discurso ejemplo",
        fecha="20240115",
        duracion=3600.5,
    )
    assert meta.video_id == "abc123"
    assert meta.url.endswith("abc123")
    assert meta.titulo == "Discurso ejemplo"
    assert meta.fecha == "20240115"
    assert meta.duracion == 3600.5


def test_video_meta_fecha_opcional() -> None:
    meta = VideoMeta(
        video_id="x",
        url="https://example.com/x",
        titulo="Sin fecha",
        fecha=None,
        duracion=10.0,
    )
    assert meta.fecha is None


def test_video_meta_es_frozen() -> None:
    meta = VideoMeta(
        video_id="x",
        url="https://example.com/x",
        titulo="T",
        fecha=None,
        duracion=1.0,
    )
    with pytest.raises(AttributeError):
        setattr(meta, "titulo", "otro")


def test_audio_video_fields(tmp_path: Path) -> None:
    wav = tmp_path / "abc123.wav"
    wav.write_bytes(b"RIFF")
    audio = AudioVideo(
        video_id="abc123",
        url="https://www.youtube.com/watch?v=abc123",
        titulo="Discurso ejemplo",
        fecha="20240115",
        audio_path=wav,
        duracion=3600.5,
    )
    assert audio.audio_path == wav
    assert audio.video_id == "abc123"
    assert audio.duracion == 3600.5


def test_audio_video_from_meta(tmp_path: Path) -> None:
    meta = VideoMeta(
        video_id="vid1",
        url="https://example.com/vid1",
        titulo="Título",
        fecha="20250101",
        duracion=120.0,
    )
    wav = tmp_path / "vid1.wav"
    wav.touch()
    audio = AudioVideo.from_meta(meta, audio_path=wav)
    assert audio.video_id == meta.video_id
    assert audio.url == meta.url
    assert audio.titulo == meta.titulo
    assert audio.fecha == meta.fecha
    assert audio.duracion == meta.duracion
    assert audio.audio_path == wav


def test_audio_video_es_frozen(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    wav.touch()
    audio = AudioVideo(
        video_id="a",
        url="https://example.com/a",
        titulo="T",
        fecha=None,
        audio_path=wav,
        duracion=1.0,
    )
    with pytest.raises(AttributeError):
        setattr(audio, "titulo", "otro")


def test_download_result_ok(tmp_path: Path) -> None:
    wav = tmp_path / "ok.wav"
    wav.touch()
    result = DownloadResult(video_id="ok", path=wav, ok=True, error=None)
    assert result.ok is True
    assert result.path == wav
    assert result.error is None


def test_download_result_error() -> None:
    result = DownloadResult(
        video_id="fail",
        path=None,
        ok=False,
        error="HTTP Error 403: Forbidden",
    )
    assert result.ok is False
    assert result.path is None
    assert "403" in (result.error or "")


def test_download_result_error_default_none(tmp_path: Path) -> None:
    wav = tmp_path / "ok.wav"
    wav.touch()
    result = DownloadResult(video_id="ok", path=wav, ok=True)
    assert result.error is None
