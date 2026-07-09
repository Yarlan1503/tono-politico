"""Tests de playlist.py (audio_fetcher)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tono_politico.speech2text.audio_fetcher.playlist import (
    obtener_info_playlist,
    sanitizar_nombre_directorio,
)


class TestSanitizarNombre:
    def test_nombre_simple(self) -> None:
        assert sanitizar_nombre_directorio("MiPlaylist") == "MiPlaylist"

    def test_espacios(self) -> None:
        assert sanitizar_nombre_directorio("Mi Playlist") == "Mi_Playlist"

    def test_caracteres_problematicos(self) -> None:
        assert sanitizar_nombre_directorio('Play:list<>?"') == "Play_list"

    def test_vacio(self) -> None:
        assert sanitizar_nombre_directorio("") == "playlist_sin_nombre"


class TestObtenerInfoPlaylist:
    def test_parseo_a_video_meta(self) -> None:
        lines = [
            json.dumps(
                {
                    "id": "vid001",
                    "title": "Video Uno",
                    "url": "https://www.youtube.com/watch?v=vid001",
                    "duration": 120.0,
                    "upload_date": "20260101",
                    "playlist": "Test Playlist",
                }
            ),
            json.dumps(
                {
                    "id": "vid002",
                    "title": "Video Dos",
                    "url": "https://www.youtube.com/watch?v=vid002",
                    "duration": 200.0,
                    "upload_date": "NA",
                    "playlist": "Test Playlist",
                }
            ),
        ]
        mock_result = MagicMock(returncode=0, stdout="\n".join(lines), stderr="")

        with patch(
            "tono_politico.speech2text.audio_fetcher.playlist.subprocess.run",
            return_value=mock_result,
        ):
            playlist, metas = obtener_info_playlist("https://youtube.com/playlist?list=FAKE")

        assert playlist.nombre == "Test_Playlist"
        assert playlist.videos == []  # VideoMeta fuera de PlaylistInfo
        assert len(metas) == 2
        assert metas[0].video_id == "vid001"
        assert metas[0].titulo == "Video Uno"
        assert metas[0].duracion == 120.0
        assert metas[0].fecha == "20260101"
        assert metas[1].fecha is None

    def test_yt_dlp_error_raises(self) -> None:
        mock_result = MagicMock(returncode=1, stdout="", stderr="boom")
        with (
            patch(
                "tono_politico.speech2text.audio_fetcher.playlist.subprocess.run",
                return_value=mock_result,
            ),
            pytest.raises(RuntimeError, match="yt-dlp falló"),
        ):
            obtener_info_playlist("https://bad")
