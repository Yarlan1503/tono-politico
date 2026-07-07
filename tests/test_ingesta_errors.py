"""Tests para DownloadResult — errores parciales como datos."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tono_politico.ingesta.audio import descargar_audio_result
from tono_politico.ingesta.models import DownloadResult
from tono_politico.models import VideoInfo


def _video(video_id: str = "abc123") -> VideoInfo:
    return VideoInfo(id=video_id, titulo="Test Video", url="https://fake", duracion=30.0)


# ──────────────────────────────────────────────────────────
# DownloadResult dataclass
# ──────────────────────────────────────────────────────────


class TestDownloadResult:
    def test_ok_true_cuando_path_existe(self):
        r = DownloadResult(video_id="v1", path=Path("/audio.wav"), ok=True)
        assert r.ok is True
        assert r.error is None

    def test_ok_false_cuando_error(self):
        r = DownloadResult(video_id="v1", path=None, ok=False, error="HTTP 403")
        assert r.ok is False
        assert r.error == "HTTP 403"

    def test_es_frozen(self):
        """Verify DownloadResult is immutable."""
        import dataclasses

        r = DownloadResult(video_id="v1", path=None, ok=False, error="x")
        assert dataclasses.is_dataclass(r)
        # frozen=True means __setattr__ raises
        # Use getattr to access the frozen __setattr__
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            r.__setattr__("ok", True)  # type: ignore[misc]


# ──────────────────────────────────────────────────────────
# descargar_audio_result — wrapper que devuelve DownloadResult
# ──────────────────────────────────────────────────────────


class TestDescargarAudioResult:
    def test_descarga_exitosa_devuelve_ok(self, tmp_path: Path):
        playlist = "test"

        def fake_run(cmd, **kwargs):
            # Simular que el archivo existe después del run
            destino = Path(cmd[cmd.index("-o") + 1])
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(b"fake audio")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("tono_politico.ingesta.audio.subprocess.run", side_effect=fake_run):
            result = descargar_audio_result(_video(), playlist, tmp_path)

        assert result.ok is True
        assert result.path is not None
        assert result.path.exists()
        assert result.error is None

    def test_timeout_devuelve_error_accionable(self, tmp_path: Path):
        with patch(
            "tono_politico.ingesta.audio.subprocess.run",
            side_effect=subprocess.TimeoutExpired("yt-dlp", 600),
        ):
            result = descargar_audio_result(_video(), "test", tmp_path)

        assert result.ok is False
        assert result.path is None
        assert result.error is not None
        assert "timeout" in result.error.lower()

    def test_returncode_non_zero_devuelve_error_truncado(self, tmp_path: Path):
        fake_result = subprocess.CompletedProcess(
            [], 1, stdout="", stderr="ERROR: [private video]\n" * 20
        )
        with patch("tono_politico.ingesta.audio.subprocess.run", return_value=fake_result):
            result = descargar_audio_result(_video(), "test", tmp_path)

        assert result.ok is False
        assert result.path is None
        assert result.error is not None
        assert len(result.error) <= 300  # truncado pero informativo
        assert "private video" in result.error or "ERROR" in result.error

    def test_postproceso_sin_archivo_devuelve_error(self, tmp_path: Path):
        fake_result = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with patch("tono_politico.ingesta.audio.subprocess.run", return_value=fake_result):
            result = descargar_audio_result(_video(), "test", tmp_path)

        assert result.ok is False
        assert result.path is None
        assert result.error is not None
        assert "no existe" in result.error.lower() or "archivo" in result.error.lower()


class TestDownloadArchive:
    def test_comando_contiene_download_archive_cuando_se_provee(self, tmp_path: Path):
        archive = tmp_path / "yt-dlp-archive.txt"
        captured_cmd: list[str] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("tono_politico.ingesta.audio.subprocess.run", side_effect=fake_run):
            descargar_audio_result(_video(), "test", tmp_path, archive_path=archive)

        assert "--download-archive" in captured_cmd
        assert str(archive) in captured_cmd

    def test_comando_no_contiene_download_archive_si_no_se_provee(self, tmp_path: Path):
        captured_cmd: list[str] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("tono_politico.ingesta.audio.subprocess.run", side_effect=fake_run):
            descargar_audio_result(_video(), "test", tmp_path)

        assert "--download-archive" not in captured_cmd
