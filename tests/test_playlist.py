"""Tests para el módulo playlist: obtener_info_playlist + sanitización."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tono_politico.ingesta.playlist import obtener_info_playlist, sanitizar_nombre_directorio

# ──────────────────────────────────────────────────────────
# Tests: sanitizar_nombre_directorio
# ──────────────────────────────────────────────────────────


class TestSanitizarNombre:
    def test_nombre_simple(self):
        assert sanitizar_nombre_directorio("MiPlaylist") == "MiPlaylist"

    def test_espacios_se_reemplazan(self):
        assert sanitizar_nombre_directorio("Mi Playlist") == "Mi_Playlist"

    def test_espacios_multiples_colapsan(self):
        assert sanitizar_nombre_directorio("Mi   Playlist") == "Mi_Playlist"

    def test_caracteres_problematicos(self):
        assert sanitizar_nombre_directorio('Play:list<>?"') == "Play_list"
        assert sanitizar_nombre_directorio("Play:List<>") == "Play_List"

    def test_strip_guiones_bajos_extremos(self):
        assert sanitizar_nombre_directorio("___Test___") == "Test"

    def test_vacio_devuelve_default(self):
        assert sanitizar_nombre_directorio("") == "playlist_sin_nombre"

    def test_solo_espacios_devuelve_default(self):
        assert sanitizar_nombre_directorio("   ") == "playlist_sin_nombre"


# ──────────────────────────────────────────────────────────
# Tests: obtener_info_playlist
# ──────────────────────────────────────────────────────────


class TestObtenerInfoPlaylist:
    def test_parseo_correcto(self, yt_dlp_json_output):
        """Verifica que el JSON de yt-dlp se parsea correctamente."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "\n".join(yt_dlp_json_output)
        mock_result.stderr = ""

        with patch("tono_politico.ingesta.playlist.subprocess.run", return_value=mock_result):
            info = obtener_info_playlist("https://youtube.com/playlist?list=FAKE")

        assert info.nombre == "TestPlaylist"
        assert len(info.videos) == 2
        assert info.videos[0].id == "vid001"
        assert info.videos[0].titulo == "Video Uno"
        assert info.videos[0].duracion == 120.0
        assert info.videos[0].fecha == "20260101"
        assert info.videos[1].id == "vid002"
        assert info.videos[1].fecha == "20260201"

    def test_fecha_na_se_convierte_none(self):
        """upload_date='NA' debe convertirse a None."""
        json_output = json.dumps(
            {
                "id": "vidX",
                "title": "Test",
                "url": "https://www.youtube.com/watch?v=vidX",
                "duration": 100.0,
                "upload_date": "NA",
                "playlist": "MiPlay",
            }
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json_output
        mock_result.stderr = ""

        with patch("tono_politico.ingesta.playlist.subprocess.run", return_value=mock_result):
            info = obtener_info_playlist("https://youtube.com/playlist?list=X")

        assert info.videos[0].fecha is None

    def test_playlist_sin_videos(self):
        """Playlist sin videos en la salida JSON."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("tono_politico.ingesta.playlist.subprocess.run", return_value=mock_result):
            info = obtener_info_playlist("https://youtube.com/playlist?list=EMPTY")

        assert info.nombre == "playlist_sin_nombre"
        assert len(info.videos) == 0

    def test_linea_json_malformada_se_ignora(self):
        """Una línea corrupta en el JSON debe loguear warning y continuar."""
        stdout = "LÍNEA CORRUPTA\n" + json.dumps(
            {
                "id": "vidOK",
                "title": "Bueno",
                "url": "https://www.youtube.com/watch?v=vidOK",
                "duration": 50.0,
                "upload_date": "20260601",
                "playlist": "TestPlay",
            }
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = stdout
        mock_result.stderr = ""

        with patch("tono_politico.ingesta.playlist.subprocess.run", return_value=mock_result):
            info = obtener_info_playlist("https://youtube.com/playlist?list=X")

        assert len(info.videos) == 1
        assert info.videos[0].id == "vidOK"

    def test_yt_dlp_falla_lanza_runtime_error(self):
        """Si yt-dlp devuelve código != 0, debe lanzar RuntimeError."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Playlist not found"

        with patch("tono_politico.ingesta.playlist.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="yt-dlp falló"):
                obtener_info_playlist("https://youtube.com/playlist?list=BAD")
