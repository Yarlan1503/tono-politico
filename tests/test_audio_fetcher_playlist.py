"""Tests de playlist.py (audio_fetcher)."""

from __future__ import annotations

import json
import subprocess
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
        assert not hasattr(playlist, "url")
        assert not hasattr(playlist, "videos")
        assert len(metas) == 2
        assert metas[0].video_id == "vid001"
        assert metas[0].titulo == "Video Uno"
        assert metas[0].duracion == 120.0
        assert metas[0].fecha == "20260101"
        assert metas[1].fecha is None

    def test_fecha_usa_release_date_si_upload_date_no_disponible(self) -> None:
        line = json.dumps(
            {
                "id": "vid-release",
                "title": "Video con release date",
                "duration": 30.0,
                "upload_date": "NA",
                "release_date": "20251231",
                "playlist": "P",
            }
        )
        mock_result = MagicMock(returncode=0, stdout=line, stderr="")

        with patch(
            "tono_politico.speech2text.audio_fetcher.playlist.subprocess.run",
            return_value=mock_result,
        ):
            _playlist, metas = obtener_info_playlist("https://youtube.com/playlist?list=P")

        assert metas[0].fecha == "20251231"

    def test_duracion_invalida_se_normaliza_a_cero(self) -> None:
        line = json.dumps(
            {
                "id": "vid-duration",
                "title": "Video con duración inválida",
                "duration": "not-a-duration",
                "playlist": "P",
            }
        )
        mock_result = MagicMock(returncode=0, stdout=line, stderr="")

        with patch(
            "tono_politico.speech2text.audio_fetcher.playlist.subprocess.run",
            return_value=mock_result,
        ):
            _playlist, metas = obtener_info_playlist("https://youtube.com/playlist?list=P")

        assert metas[0].duracion == 0.0

    def test_yt_dlp_timeout_raises_runtime_error(self) -> None:
        with (
            patch(
                "tono_politico.speech2text.audio_fetcher.playlist.subprocess.run",
                side_effect=subprocess.TimeoutExpired("yt-dlp", 120),
            ),
            pytest.raises(RuntimeError, match="timeout"),
        ):
            obtener_info_playlist("https://slow")

    def test_yt_dlp_ausente_raises_runtime_error(self) -> None:
        with (
            patch(
                "tono_politico.speech2text.audio_fetcher.playlist.subprocess.run",
                side_effect=FileNotFoundError("yt-dlp"),
            ),
            pytest.raises(RuntimeError, match="yt-dlp"),
        ):
            obtener_info_playlist("https://missing-binary")

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

    def test_lineas_invalidas_y_json_no_objeto_se_ignoran(self) -> None:
        valid = json.dumps(
            {
                "id": "valid",
                "title": "Válido",
                "playlist": "P",
                "duration": 10,
            }
        )
        mock_result = MagicMock(
            returncode=0,
            stdout="not-json\n[]\n" + valid,
            stderr="",
        )

        with patch(
            "tono_politico.speech2text.audio_fetcher.playlist.subprocess.run",
            return_value=mock_result,
        ):
            _playlist, metas = obtener_info_playlist("https://youtube.com/playlist?list=P")

        assert [meta.video_id for meta in metas] == ["valid"]

    def test_registro_sin_id_se_ignora(self) -> None:
        line = json.dumps({"title": "Sin ID", "playlist": "P", "duration": 10})
        mock_result = MagicMock(returncode=0, stdout=line, stderr="")

        with patch(
            "tono_politico.speech2text.audio_fetcher.playlist.subprocess.run",
            return_value=mock_result,
        ):
            _playlist, metas = obtener_info_playlist("https://youtube.com/playlist?list=P")

        assert metas == []

    def test_fecha_invalida_se_expone_como_ausente(self) -> None:
        line = json.dumps(
            {
                "id": "invalid-date",
                "title": "Fecha inválida",
                "playlist": "P",
                "upload_date": "20261399",
                "release_date": "NA",
            }
        )
        mock_result = MagicMock(returncode=0, stdout=line, stderr="")

        with patch(
            "tono_politico.speech2text.audio_fetcher.playlist.subprocess.run",
            return_value=mock_result,
        ):
            _playlist, metas = obtener_info_playlist("https://youtube.com/playlist?list=P")

        assert metas[0].fecha is None
