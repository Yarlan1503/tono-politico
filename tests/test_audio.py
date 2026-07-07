"""Tests para el módulo audio: verificar_cache_videos + descargar_audio."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tono_politico.ingesta.audio import descargar_audio, verificar_cache_videos

# ──────────────────────────────────────────────────────────
# Tests: verificar_cache_videos
# ──────────────────────────────────────────────────────────


class TestVerificarCacheVideos:
    def test_carpeta_inexistente_todo_faltante(self, playlist_mock, tmp_path):
        """Si no existe la carpeta de videos, todo debe ser faltante."""
        estado = verificar_cache_videos(
            playlist_mock.nombre, playlist_mock.videos, base_dir=tmp_path
        )

        assert len(estado["existentes"]) == 0
        assert len(estado["faltantes"]) == 3

    def test_carpeta_vacia_todo_faltante(self, playlist_mock, tmp_path):
        """Si la carpeta existe pero está vacía, todo es faltante."""
        dir_videos = tmp_path / playlist_mock.nombre / f"videos-{playlist_mock.nombre}"
        dir_videos.mkdir(parents=True)

        estado = verificar_cache_videos(
            playlist_mock.nombre, playlist_mock.videos, base_dir=tmp_path
        )

        assert len(estado["existentes"]) == 0
        assert len(estado["faltantes"]) == 3

    def test_un_parcial_faltan_dos(self, playlist_mock, tmp_path):
        """Si hay 1 de 3 audios, debe detectar 1 existente y 2 faltantes."""
        dir_videos = tmp_path / playlist_mock.nombre / f"videos-{playlist_mock.nombre}"
        dir_videos.mkdir(parents=True)
        (dir_videos / "vid001.wav").write_bytes(b"fake_audio")

        estado = verificar_cache_videos(
            playlist_mock.nombre, playlist_mock.videos, base_dir=tmp_path
        )

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

        estado = verificar_cache_videos(
            playlist_mock.nombre, playlist_mock.videos, base_dir=tmp_path
        )

        assert len(estado["existentes"]) == 3
        assert len(estado["faltantes"]) == 0

    def test_playlist_vacia(self, playlist_vacia_mock, tmp_path):
        """Playlist sin videos debe devolver todo vacío."""
        estado = verificar_cache_videos(
            playlist_vacia_mock.nombre,
            playlist_vacia_mock.videos,
            base_dir=tmp_path,
        )

        assert len(estado["existentes"]) == 0
        assert len(estado["faltantes"]) == 0

    def test_video_en_cache_no_en_playlist_se_ignora(self, playlist_mock, tmp_path):
        """Un archivo .wav en cache que no corresponde a ningún video se ignora."""
        dir_videos = tmp_path / playlist_mock.nombre / f"videos-{playlist_mock.nombre}"
        dir_videos.mkdir(parents=True)
        (dir_videos / "vid001.wav").write_bytes(b"fake")
        (dir_videos / "vid999.wav").write_bytes(b"fake")

        estado = verificar_cache_videos(
            playlist_mock.nombre, playlist_mock.videos, base_dir=tmp_path
        )

        assert len(estado["existentes"]) == 1
        assert len(estado["faltantes"]) == 2

    def test_mantiene_orden_de_playlist(self, playlist_mock, tmp_path):
        """El orden de las listas debe respetar el orden de la playlist."""
        dir_videos = tmp_path / playlist_mock.nombre / f"videos-{playlist_mock.nombre}"
        dir_videos.mkdir(parents=True)
        for v in playlist_mock.videos:
            (dir_videos / f"{v.id}.wav").write_bytes(b"fake")

        estado = verificar_cache_videos(
            playlist_mock.nombre, playlist_mock.videos, base_dir=tmp_path
        )

        existentes_ids = [v.id for v in estado["existentes"]]
        assert existentes_ids == ["vid001", "vid002", "vid003"]


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

        with patch("tono_politico.ingesta.audio.subprocess.run") as mock_run:

            def side_effect(*args, **kwargs):
                ruta = tmp_path / "TestPlaylist" / (f"videos-TestPlaylist/{video.id}.wav")
                ruta.parent.mkdir(parents=True, exist_ok=True)
                ruta.write_bytes(b"fake_audio_data")
                return mock_result

            mock_run.side_effect = side_effect
            ruta = descargar_audio(video, playlist_mock.nombre, base_dir=tmp_path)

        assert ruta is not None
        assert ruta.exists()
        assert ruta.name == f"{video.id}.wav"
        assert "videos-TestPlaylist" in str(ruta)

    def test_devuelve_none_si_yt_dlp_falla(self, playlist_mock, tmp_path):
        """Si yt-dlp falla, devuelve None (graceful degradation)."""
        video = playlist_mock.videos[0]
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Video unavailable"

        with patch(
            "tono_politico.ingesta.audio.subprocess.run",
            return_value=mock_result,
        ):
            resultado = descargar_audio(video, playlist_mock.nombre, base_dir=tmp_path)

        assert resultado is None

    def test_devuelve_none_si_archivo_no_creado(self, playlist_mock, tmp_path):
        """Si yt-dlp termina OK pero el archivo no existe, devuelve None."""
        video = playlist_mock.videos[0]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch(
            "tono_politico.ingesta.audio.subprocess.run",
            return_value=mock_result,
        ):
            resultado = descargar_audio(video, playlist_mock.nombre, base_dir=tmp_path)

        assert resultado is None
