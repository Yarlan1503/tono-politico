"""Tests para el Componente 1: Ingesta — métodos 1-4.

Cobertura:
- obtener_info_playlist: playlist inválida, parseo correcto
- verificar_cache_videos: carpeta inexistente, parcial, completa
- verificar_cache_transcripciones: carpeta inexistente, parcial, completa, JSON inválido
- descargar_audio: estructura de carpetas
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tono_politico.ingesta.playlist import (
    DATA_DIR,
    _sanitizar_nombre_directorio,
    descargar_audio,
    obtener_info_playlist,
    verificar_cache_transcripciones,
    verificar_cache_videos,
)
from tono_politico.models import PlaylistInfo, VideoInfo


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture
def playlist_mock() -> PlaylistInfo:
    """Playlist simulada con 3 videos de prueba."""
    return PlaylistInfo(
        nombre="TestPlaylist",
        url="https://youtube.com/playlist?list=FAKE",
        videos=[
            VideoInfo(id="vid001", titulo="Video Uno", url="https://www.youtube.com/watch?v=vid001", duracion=120.0, fecha="20260101"),
            VideoInfo(id="vid002", titulo="Video Dos", url="https://www.youtube.com/watch?v=vid002", duracion=300.0, fecha="20260201"),
            VideoInfo(id="vid003", titulo="Video Tres", url="https://www.youtube.com/watch?v=vid003", duracion=60.0, fecha="20260301"),
        ],
    )


@pytest.fixture
def playlist_vacia_mock() -> PlaylistInfo:
    """Playlist simulada sin videos."""
    return PlaylistInfo(
        nombre="PlaylistVacia",
        url="https://youtube.com/playlist?list=EMPTY",
        videos=[],
    )


@pytest.fixture
def yt_dlp_json_output() -> list[str]:
    """Simula la salida JSON línea-por-línea de yt-dlp --flat-playlist -j."""
    return [
        json.dumps({
            "_type": "url",
            "ie_key": "Youtube",
            "id": "vid001",
            "title": "Video Uno",
            "url": "https://www.youtube.com/watch?v=vid001",
            "duration": 120.0,
            "upload_date": "20260101",
            "playlist": "TestPlaylist",
        }),
        json.dumps({
            "_type": "url",
            "ie_key": "Youtube",
            "id": "vid002",
            "title": "Video Dos",
            "url": "https://www.youtube.com/watch?v=vid002",
            "duration": 300.0,
            "upload_date": "20260201",
            "playlist": "TestPlaylist",
        }),
    ]


# ──────────────────────────────────────────────────────────
# Tests: _sanitizar_nombre_directorio
# ──────────────────────────────────────────────────────────

class TestSanitizarNombre:
    def test_nombre_simple(self):
        assert _sanitizar_nombre_directorio("MiPlaylist") == "MiPlaylist"

    def test_espacios_se_reemplazan(self):
        assert _sanitizar_nombre_directorio("Mi Playlist") == "Mi_Playlist"

    def test_espacios_multiples_colapsan(self):
        assert _sanitizar_nombre_directorio("Mi   Playlist") == "Mi_Playlist"

    def test_caracteres_problematicos(self):
        # Los caracteres problemáticos se reemplazan por _ pero los _ finales se eliminan con strip
        assert _sanitizar_nombre_directorio('Play:list<>?"') == "Play_list"
        # Pero un _ en el medio se preserva
        assert _sanitizar_nombre_directorio('Play:List<>') == "Play_List"

    def test_strip_guiones_bajos_extremos(self):
        assert _sanitizar_nombre_directorio("___Test___") == "Test"

    def test_vacio_devuelve_default(self):
        assert _sanitizar_nombre_directorio("") == "playlist_sin_nombre"

    def test_solo_espacios_devuelve_default(self):
        assert _sanitizar_nombre_directorio("   ") == "playlist_sin_nombre"


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
        json_output = json.dumps({
            "id": "vidX",
            "title": "Test",
            "url": "https://www.youtube.com/watch?v=vidX",
            "duration": 100.0,
            "upload_date": "NA",
            "playlist": "MiPlay",
        })
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
        stdout = "LÍNEA CORRUPTA\n" + json.dumps({
            "id": "vidOK",
            "title": "Bueno",
            "url": "https://www.youtube.com/watch?v=vidOK",
            "duration": 50.0,
            "upload_date": "20260601",
            "playlist": "TestPlay",
        })
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


# ──────────────────────────────────────────────────────────
# Tests: verificar_cache_videos
# ──────────────────────────────────────────────────────────

class TestVerificarCacheVideos:
    def test_carpeta_inexistente_todo_faltante(self, playlist_mock, tmp_path):
        """Si no existe la carpeta de videos, todo debe ser faltante."""
        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path):
            estado = verificar_cache_videos(playlist_mock.nombre, playlist_mock.videos)

        assert len(estado["existentes"]) == 0
        assert len(estado["faltantes"]) == 3

    def test_carpeta_vacia_todo_faltante(self, playlist_mock, tmp_path):
        """Si la carpeta existe pero está vacía, todo es faltante."""
        dir_videos = tmp_path / playlist_mock.nombre / f"videos-{playlist_mock.nombre}"
        dir_videos.mkdir(parents=True)

        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path):
            estado = verificar_cache_videos(playlist_mock.nombre, playlist_mock.videos)

        assert len(estado["existentes"]) == 0
        assert len(estado["faltantes"]) == 3

    def test_un_parcial_faltan_dos(self, playlist_mock, tmp_path):
        """Si hay 1 de 3 audios, debe detectar 1 existente y 2 faltantes."""
        dir_videos = tmp_path / playlist_mock.nombre / f"videos-{playlist_mock.nombre}"
        dir_videos.mkdir(parents=True)
        (dir_videos / "vid001.wav").write_bytes(b"fake_audio")

        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path):
            estado = verificar_cache_videos(playlist_mock.nombre, playlist_mock.videos)

        assert len(estado["existentes"]) == 1
        assert estado["existentes"][0].id == "vid001"
        assert len(estado["faltantes"]) == 2
        faltantes_ids = {v.id for v in estado["faltantes"]}
        assert faltantes_ids == {"vid002", "vid003"}

    def test_todos_descargados_nada_faltante(self, playlist_mock, tmp_path):
        """Si todos los audios existen, nada es faltante."""
        dir_videos = tmp_path / playlist_mock.nombre / f"videos-{playlist_mock.nombre}"
        dir_videos.mkdir(parents=True)
        for v in playlist_mock.videos:
            (dir_videos / f"{v.id}.wav").write_bytes(b"fake_audio")

        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path):
            estado = verificar_cache_videos(playlist_mock.nombre, playlist_mock.videos)

        assert len(estado["existentes"]) == 3
        assert len(estado["faltantes"]) == 0

    def test_playlist_vacia(self, playlist_vacia_mock, tmp_path):
        """Playlist sin videos debe devolver todo vacío."""
        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path):
            estado = verificar_cache_videos(playlist_vacia_mock.nombre, playlist_vacia_mock.videos)

        assert len(estado["existentes"]) == 0
        assert len(estado["faltantes"]) == 0

    def test_video_en_cache_no_en_playlist_se_ignora(self, playlist_mock, tmp_path):
        """Un archivo .wav en cache que no corresponde a ningún video se ignora."""
        dir_videos = tmp_path / playlist_mock.nombre / f"videos-{playlist_mock.nombre}"
        dir_videos.mkdir(parents=True)
        (dir_videos / "vid001.wav").write_bytes(b"fake")
        (dir_videos / "vid999.wav").write_bytes(b"fake")  # No está en la playlist

        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path):
            estado = verificar_cache_videos(playlist_mock.nombre, playlist_mock.videos)

        assert len(estado["existentes"]) == 1  # Solo vid001
        assert len(estado["faltantes"]) == 2

    def test_mantiene_orden_de_playlist(self, playlist_mock, tmp_path):
        """El orden de las listas debe respetar el orden de la playlist."""
        dir_videos = tmp_path / playlist_mock.nombre / f"videos-{playlist_mock.nombre}"
        dir_videos.mkdir(parents=True)
        for v in playlist_mock.videos:
            (dir_videos / f"{v.id}.wav").write_bytes(b"fake")

        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path):
            estado = verificar_cache_videos(playlist_mock.nombre, playlist_mock.videos)

        existentes_ids = [v.id for v in estado["existentes"]]
        assert existentes_ids == ["vid001", "vid002", "vid003"]


# ──────────────────────────────────────────────────────────
# Tests: verificar_cache_transcripciones
# ──────────────────────────────────────────────────────────

class TestVerificarCacheTranscripciones:
    def test_carpeta_inexistente_todo_faltante(self, playlist_mock, tmp_path):
        """Si no existe transcripciones-<playlist>/, todo debe ser faltante."""
        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path):
            estado = verificar_cache_transcripciones(playlist_mock.nombre, playlist_mock.videos)

        assert len(estado["existentes"]) == 0
        assert len(estado["faltantes"]) == 3

    def test_carpeta_vacia_todo_faltante(self, playlist_mock, tmp_path):
        """Si la carpeta existe pero está vacía, todo es faltante."""
        dir_transcripciones = tmp_path / playlist_mock.nombre / f"transcripciones-{playlist_mock.nombre}"
        dir_transcripciones.mkdir(parents=True)

        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path):
            estado = verificar_cache_transcripciones(playlist_mock.nombre, playlist_mock.videos)

        assert len(estado["existentes"]) == 0
        assert len(estado["faltantes"]) == 3

    def test_cache_parcial_detecta_faltantes(self, playlist_mock, tmp_path):
        """Si existe 1 de 3 JSONs, detecta 1 existente y 2 faltantes."""
        dir_transcripciones = tmp_path / playlist_mock.nombre / f"transcripciones-{playlist_mock.nombre}"
        dir_transcripciones.mkdir(parents=True)
        (dir_transcripciones / "vid001.json").write_text(
            json.dumps({"video_id": "vid001", "raw_segments": []}), encoding="utf-8"
        )

        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path):
            estado = verificar_cache_transcripciones(playlist_mock.nombre, playlist_mock.videos)

        assert len(estado["existentes"]) == 1
        assert estado["existentes"][0].id == "vid001"
        assert {v.id for v in estado["faltantes"]} == {"vid002", "vid003"}

    def test_todos_los_jsons_validos_nada_faltante(self, playlist_mock, tmp_path):
        """Si todos los JSONs existen y son válidos, nada es faltante."""
        dir_transcripciones = tmp_path / playlist_mock.nombre / f"transcripciones-{playlist_mock.nombre}"
        dir_transcripciones.mkdir(parents=True)
        for video in playlist_mock.videos:
            (dir_transcripciones / f"{video.id}.json").write_text(
                json.dumps({"video_id": video.id, "raw_segments": []}), encoding="utf-8"
            )

        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path):
            estado = verificar_cache_transcripciones(playlist_mock.nombre, playlist_mock.videos)

        assert len(estado["existentes"]) == 3
        assert len(estado["faltantes"]) == 0

    def test_json_corrupto_cuenta_como_faltante(self, playlist_mock, tmp_path):
        """Un JSON corrupto no debe considerarse transcripción existente."""
        dir_transcripciones = tmp_path / playlist_mock.nombre / f"transcripciones-{playlist_mock.nombre}"
        dir_transcripciones.mkdir(parents=True)
        (dir_transcripciones / "vid001.json").write_text("{json roto", encoding="utf-8")

        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path):
            estado = verificar_cache_transcripciones(playlist_mock.nombre, playlist_mock.videos)

        assert len(estado["existentes"]) == 0
        assert {v.id for v in estado["faltantes"]} == {"vid001", "vid002", "vid003"}

    def test_json_con_video_id_distinto_cuenta_como_faltante(self, playlist_mock, tmp_path):
        """Un JSON cuyo video_id no coincide evita reutilizar cache equivocado."""
        dir_transcripciones = tmp_path / playlist_mock.nombre / f"transcripciones-{playlist_mock.nombre}"
        dir_transcripciones.mkdir(parents=True)
        (dir_transcripciones / "vid001.json").write_text(
            json.dumps({"video_id": "otro_id", "raw_segments": []}), encoding="utf-8"
        )

        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path):
            estado = verificar_cache_transcripciones(playlist_mock.nombre, playlist_mock.videos)

        assert len(estado["existentes"]) == 0
        assert {v.id for v in estado["faltantes"]} == {"vid001", "vid002", "vid003"}


# ──────────────────────────────────────────────────────────
# Tests: descargar_audio (estructura de carpetas)
# ──────────────────────────────────────────────────────────

class TestDescargarAudio:
    def test_crea_estructura_carpetas(self, playlist_mock, tmp_path):
        """descargar_audio debe crear videos-<nombre>/ aunque no exista."""
        video = playlist_mock.videos[0]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path), \
             patch("tono_politico.ingesta.playlist.subprocess.run") as mock_run:
            # Simular que yt-dlp crea el archivo
            def side_effect(*args, **kwargs):
                # El comando real crearía el archivo; lo simulamos
                ruta = tmp_path / video.nombre if hasattr(video, 'nombre') else (tmp_path / "TestPlaylist" / "videos-TestPlaylist" / f"{video.id}.wav")
                ruta.parent.mkdir(parents=True, exist_ok=True)
                ruta.write_bytes(b"fake_audio_data")
                return mock_result

            mock_run.side_effect = side_effect
            ruta = descargar_audio(video, playlist_mock.nombre)

        assert ruta.exists()
        assert ruta.name == f"{video.id}.wav"
        assert "videos-TestPlaylist" in str(ruta)

    def test_lanza_error_si_yt_dlp_falla(self, playlist_mock, tmp_path):
        """Si yt-dlp falla, debe lanzar RuntimeError."""
        video = playlist_mock.videos[0]
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Video unavailable"

        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path), \
             patch("tono_politico.ingesta.playlist.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Error descargando video"):
                descargar_audio(video, playlist_mock.nombre)

    def test_lanza_error_si_archivo_no_creado(self, playlist_mock, tmp_path):
        """Si yt-dlp termina OK pero el archivo no existe, debe lanzar RuntimeError."""
        video = playlist_mock.videos[0]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch("tono_politico.ingesta.playlist.DATA_DIR", tmp_path), \
             patch("tono_politico.ingesta.playlist.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="el archivo no existe"):
                descargar_audio(video, playlist_mock.nombre)
