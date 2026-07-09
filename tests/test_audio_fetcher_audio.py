"""Tests de audio.py (cache + descarga)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from tono_politico.speech2text.audio_fetcher.audio import (
    descargar_audio_result,
    verificar_cache_videos,
)
from tono_politico.speech2text.audio_fetcher.models import VideoMeta


def _meta(video_id: str = "vid001") -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        titulo=f"Título {video_id}",
        fecha="20260101",
        duracion=30.0,
    )


class TestVerificarCacheVideos:
    def test_carpeta_inexistente(self, tmp_path: Path) -> None:
        videos = [_meta("a"), _meta("b")]
        estado = verificar_cache_videos("P", videos, tmp_path)
        assert estado["existentes"] == []
        assert len(estado["faltantes"]) == 2

    def test_parcial(self, tmp_path: Path) -> None:
        dir_videos = tmp_path / "P" / "videos-P"
        dir_videos.mkdir(parents=True)
        (dir_videos / "a.wav").write_bytes(b"x")
        videos = [_meta("a"), _meta("b")]
        estado = verificar_cache_videos("P", videos, tmp_path)
        assert [v.video_id for v in estado["existentes"]] == ["a"]
        assert [v.video_id for v in estado["faltantes"]] == ["b"]


class TestDescargarAudioResult:
    def test_ok(self, tmp_path: Path) -> None:
        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            destino = Path(cmd[cmd.index("-o") + 1])
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(b"fake")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch(
            "tono_politico.speech2text.audio_fetcher.audio.subprocess.run",
            side_effect=fake_run,
        ):
            result = descargar_audio_result(_meta(), "P", tmp_path)

        assert result.ok is True
        assert result.path is not None
        assert result.path.exists()

    def test_timeout(self, tmp_path: Path) -> None:
        with patch(
            "tono_politico.speech2text.audio_fetcher.audio.subprocess.run",
            side_effect=subprocess.TimeoutExpired("yt-dlp", 600),
        ):
            result = descargar_audio_result(_meta(), "P", tmp_path)
        assert result.ok is False
        assert result.path is None
        assert "Timeout" in (result.error or "")

    def test_stderr_error(self, tmp_path: Path) -> None:
        with patch(
            "tono_politico.speech2text.audio_fetcher.audio.subprocess.run",
            return_value=subprocess.CompletedProcess(
                ["yt-dlp"], 1, stdout="", stderr="HTTP Error 403"
            ),
        ):
            result = descargar_audio_result(_meta(), "P", tmp_path)
        assert result.ok is False
        assert "403" in (result.error or "")
