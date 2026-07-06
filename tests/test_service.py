"""Tests para IngestaService: el service OOP del Componente 1."""

from __future__ import annotations

from unittest.mock import patch

from tono_politico.ingesta.service import IngestaService
from tono_politico.ingesta.transcripcion import guardar_transcripcion
from tono_politico.models import VideoTranscript

# Atajos para rutas de patch
_S = "tono_politico.ingesta.service."


class TestIngestaService:
    def test_init_guarda_config(self):
        """Los parámetros del constructor se guardan en self."""
        from pathlib import Path

        svc = IngestaService(
            data_dir=Path("/tmp/test"),
            whisper_model="tiny",
            idioma="en",
        )
        assert str(svc.data_dir) == "/tmp/test"
        assert svc.whisper_model == "tiny"
        assert svc.idioma == "en"

    def test_init_defaults(self):
        """Los defaults del constructor son correctos."""
        svc = IngestaService()
        assert svc.data_dir.name == "data"
        assert svc.whisper_model == "large-v3-turbo"
        assert svc.idioma == "es"

    def test_todo_desde_cache(self, playlist_mock, tmp_path):
        """Si todas las transcripciones están en cache, no descarga ni transcribe."""
        svc = IngestaService(data_dir=tmp_path)

        # Pre-poblar cache con transcripciones válidas
        for video in playlist_mock.videos:
            vt = VideoTranscript(
                video_id=video.id,
                url=video.url,
                titulo=video.titulo,
                fecha=video.fecha,
                raw_segments=[],
            )
            guardar_transcripcion(vt, playlist_mock.nombre, base_dir=tmp_path)

        with (
            patch(_S + "obtener_info_playlist", return_value=playlist_mock),
            patch(_S + "descargar_audio") as mock_descargar,
            patch(_S + "transcribir") as mock_transcribir,
        ):
            resultados = svc.procesar("https://fake/url")

        assert len(resultados) == 3
        assert {r.video_id for r in resultados} == {"vid001", "vid002", "vid003"}
        mock_descargar.assert_not_called()
        mock_transcribir.assert_not_called()

    def test_todo_desde_cero(self, playlist_mock, transcript_mock, tmp_path):
        """Sin cache: descarga audio, transcribe, guarda y devuelve resultados."""
        svc = IngestaService(data_dir=tmp_path)

        with (
            patch(_S + "obtener_info_playlist", return_value=playlist_mock),
            patch(_S + "descargar_audio") as mock_descargar,
            patch(
                _S + "transcribir",
                return_value=transcript_mock.raw_segments,
            ) as mock_transcribir,
        ):
            resultados = svc.procesar("https://fake/url")

        assert len(resultados) == 3
        assert mock_descargar.call_count == 3
        assert mock_transcribir.call_count == 3
        assert resultados[0].raw_segments == transcript_mock.raw_segments
        # Verificar que se guardaron en disco
        dir_t = tmp_path / playlist_mock.nombre / (
            f"transcripciones-{playlist_mock.nombre}"
        )
        for video in playlist_mock.videos:
            assert (dir_t / f"{video.id}.json").exists()

    def test_cache_parcial(self, playlist_mock, transcript_mock, tmp_path):
        """Si 1 de 3 está en cache, solo procesa los 2 faltantes."""
        svc = IngestaService(data_dir=tmp_path)

        # Pre-poblar solo vid001
        guardar_transcripcion(
            VideoTranscript(
                video_id="vid001",
                url="https://www.youtube.com/watch?v=vid001",
                titulo="Video Uno",
                fecha="20260101",
                raw_segments=[],
            ),
            playlist_mock.nombre,
            base_dir=tmp_path,
        )

        with (
            patch(_S + "obtener_info_playlist", return_value=playlist_mock),
            patch(_S + "descargar_audio") as mock_descargar,
            patch(
                _S + "transcribir",
                return_value=transcript_mock.raw_segments,
            ),
        ):
            resultados = svc.procesar("https://fake/url")

        assert len(resultados) == 3
        assert mock_descargar.call_count == 2
        transcribed_video_ids = {
            call.args[0].id for call in mock_descargar.call_args_list
        }
        assert "vid001" not in transcribed_video_ids

    def test_playlist_vacia_devuelve_lista_vacia(
        self, playlist_vacia_mock, tmp_path
    ):
        """Playlist sin videos devuelve lista vacía sin errores."""
        svc = IngestaService(data_dir=tmp_path)

        with patch(_S + "obtener_info_playlist", return_value=playlist_vacia_mock):
            resultados = svc.procesar("https://fake/empty")

        assert resultados == []

    def test_respeta_orden_de_playlist(self, playlist_mock, transcript_mock, tmp_path):
        """Los resultados deben estar en el mismo orden que la playlist."""
        svc = IngestaService(data_dir=tmp_path)

        with (
            patch(_S + "obtener_info_playlist", return_value=playlist_mock),
            patch(_S + "descargar_audio"),
            patch(
                _S + "transcribir",
                return_value=transcript_mock.raw_segments,
            ),
        ):
            resultados = svc.procesar("https://fake/url")

        ids_resultado = [r.video_id for r in resultados]
        ids_playlist = [v.id for v in playlist_mock.videos]
        assert ids_resultado == ids_playlist

    def test_cumple_componente_protocol(self):
        """IngestaService cumple el ComponenteProtocol (duck typing)."""
        from tono_politico.protocol import ComponenteProtocol

        svc = IngestaService()
        assert isinstance(svc, ComponenteProtocol)
