"""Tests de AudioFetcherService."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from tono_politico.speech2text.audio_fetcher.models import PlaylistInfo, VideoMeta
from tono_politico.speech2text.audio_fetcher.service import AudioFetcherService


def _meta(video_id: str = "vid001") -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        titulo=f"T {video_id}",
        fecha=None,
        duracion=10.0,
    )


class TestAudioFetcherService:
    def test_discover_delega(self, tmp_path: Path) -> None:
        svc = AudioFetcherService(data_dir=tmp_path)
        playlist = PlaylistInfo(nombre="P", url="u", videos=[])
        metas = [_meta()]
        with patch(
            "tono_politico.speech2text.audio_fetcher.service.obtener_info_playlist",
            return_value=(playlist, metas),
        ) as mock_discover:
            out_p, out_m = svc.discover("https://playlist")
        mock_discover.assert_called_once_with("https://playlist")
        assert out_p.nombre == "P"
        assert out_m == metas

    def test_fetch_one_usa_cache(self, tmp_path: Path) -> None:
        svc = AudioFetcherService(data_dir=tmp_path)
        meta = _meta("cached")
        wav = tmp_path / "P" / "videos-P" / "cached.wav"
        wav.parent.mkdir(parents=True)
        wav.write_bytes(b"audio")

        audio = svc.fetch_one(meta, "P")
        assert audio is not None
        assert audio.audio_path == wav
        assert audio.video_id == "cached"

    def test_fetch_one_descarga(self, tmp_path: Path) -> None:
        svc = AudioFetcherService(data_dir=tmp_path)
        meta = _meta("new")

        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            destino = Path(cmd[cmd.index("-o") + 1])
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(b"dl")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch(
            "tono_politico.speech2text.audio_fetcher.audio.subprocess.run",
            side_effect=fake_run,
        ):
            audio = svc.fetch_one(meta, "P")

        assert audio is not None
        assert audio.audio_path.exists()
        assert audio.titulo == meta.titulo

    def test_fetch_one_falla_devuelve_none(self, tmp_path: Path) -> None:
        svc = AudioFetcherService(data_dir=tmp_path)
        with patch(
            "tono_politico.speech2text.audio_fetcher.audio.subprocess.run",
            return_value=subprocess.CompletedProcess(["yt-dlp"], 1, stdout="", stderr="fail"),
        ):
            assert svc.fetch_one(_meta("x"), "P") is None

    def test_procesar_wrapper(self, tmp_path: Path) -> None:
        svc = AudioFetcherService(data_dir=tmp_path)
        playlist = PlaylistInfo(nombre="P", url="u", videos=[])
        metas = [_meta("a"), _meta("b")]
        wav_a = tmp_path / "P" / "videos-P" / "a.wav"
        wav_a.parent.mkdir(parents=True)
        wav_a.write_bytes(b"a")

        def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            destino = Path(cmd[cmd.index("-o") + 1])
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(b"b")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch(
                "tono_politico.speech2text.audio_fetcher.service.obtener_info_playlist",
                return_value=(playlist, metas),
            ),
            patch(
                "tono_politico.speech2text.audio_fetcher.audio.subprocess.run",
                side_effect=fake_run,
            ),
        ):
            audios = svc.procesar("https://playlist")

        assert len(audios) == 2
        assert {a.video_id for a in audios} == {"a", "b"}
