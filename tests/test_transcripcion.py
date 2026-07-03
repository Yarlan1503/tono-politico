"""Tests para el módulo transcripcion: transcribir, persistencia y cache."""

from __future__ import annotations

import json
import sys

import pytest

from tono_politico.ingesta.transcripcion import (
    cargar_transcripcion,
    guardar_transcripcion,
    transcribir,
    verificar_cache_transcripciones,
)
from tono_politico.models import SegmentoRaw, WordTimestamp

# ──────────────────────────────────────────────────────────
# Fakes de Whisper (solo para test_transcribir)
# ──────────────────────────────────────────────────────────

class FakeWhisperModel:
    def __init__(self, resultado):
        self.resultado = resultado
        self.transcribe_calls = []

    def transcribe(self, audio_path, **kwargs):
        self.transcribe_calls.append({"audio_path": audio_path, "kwargs": kwargs})
        return self.resultado


class FakeWhisperModule:
    def __init__(self, resultado):
        self.modelo_cargado = None
        self.model = FakeWhisperModel(resultado)

    def load_model(self, modelo):
        self.modelo_cargado = modelo
        return self.model


# ──────────────────────────────────────────────────────────
# Tests: verificar_cache_transcripciones
# ──────────────────────────────────────────────────────────

class TestVerificarCacheTranscripciones:
    def test_carpeta_inexistente_todo_faltante(self, playlist_mock, tmp_path):
        estado = verificar_cache_transcripciones(
            playlist_mock.nombre, playlist_mock.videos, base_dir=tmp_path
        )

        assert len(estado["existentes"]) == 0
        assert len(estado["faltantes"]) == 3

    def test_carpeta_vacia_todo_faltante(self, playlist_mock, tmp_path):
        dir_t = tmp_path / playlist_mock.nombre / (
            f"transcripciones-{playlist_mock.nombre}"
        )
        dir_t.mkdir(parents=True)

        estado = verificar_cache_transcripciones(
            playlist_mock.nombre, playlist_mock.videos, base_dir=tmp_path
        )

        assert len(estado["existentes"]) == 0
        assert len(estado["faltantes"]) == 3

    def test_cache_parcial_detecta_faltantes(self, playlist_mock, tmp_path):
        dir_t = tmp_path / playlist_mock.nombre / (
            f"transcripciones-{playlist_mock.nombre}"
        )
        dir_t.mkdir(parents=True)
        (dir_t / "vid001.json").write_text(
            json.dumps({"video_id": "vid001", "raw_segments": []}),
            encoding="utf-8",
        )

        estado = verificar_cache_transcripciones(
            playlist_mock.nombre, playlist_mock.videos, base_dir=tmp_path
        )

        assert len(estado["existentes"]) == 1
        assert estado["existentes"][0].id == "vid001"
        assert {v.id for v in estado["faltantes"]} == {"vid002", "vid003"}

    def test_todos_los_jsons_validos_nada_faltante(self, playlist_mock, tmp_path):
        dir_t = tmp_path / playlist_mock.nombre / (
            f"transcripciones-{playlist_mock.nombre}"
        )
        dir_t.mkdir(parents=True)
        for video in playlist_mock.videos:
            (dir_t / f"{video.id}.json").write_text(
                json.dumps({"video_id": video.id, "raw_segments": []}),
                encoding="utf-8",
            )

        estado = verificar_cache_transcripciones(
            playlist_mock.nombre, playlist_mock.videos, base_dir=tmp_path
        )

        assert len(estado["existentes"]) == 3
        assert len(estado["faltantes"]) == 0

    def test_json_corrupto_cuenta_como_faltante(self, playlist_mock, tmp_path):
        dir_t = tmp_path / playlist_mock.nombre / (
            f"transcripciones-{playlist_mock.nombre}"
        )
        dir_t.mkdir(parents=True)
        (dir_t / "vid001.json").write_text("{json roto", encoding="utf-8")

        estado = verificar_cache_transcripciones(
            playlist_mock.nombre, playlist_mock.videos, base_dir=tmp_path
        )

        assert len(estado["existentes"]) == 0
        assert {v.id for v in estado["faltantes"]} == {
            "vid001", "vid002", "vid003",
        }

    def test_json_con_video_id_distinto_cuenta_como_faltante(
        self, playlist_mock, tmp_path
    ):
        dir_t = tmp_path / playlist_mock.nombre / (
            f"transcripciones-{playlist_mock.nombre}"
        )
        dir_t.mkdir(parents=True)
        (dir_t / "vid001.json").write_text(
            json.dumps({"video_id": "otro_id", "raw_segments": []}),
            encoding="utf-8",
        )

        estado = verificar_cache_transcripciones(
            playlist_mock.nombre, playlist_mock.videos, base_dir=tmp_path
        )

        assert len(estado["existentes"]) == 0
        assert {v.id for v in estado["faltantes"]} == {
            "vid001", "vid002", "vid003",
        }


# ──────────────────────────────────────────────────────────
# Tests: transcribir
# ──────────────────────────────────────────────────────────

class TestTranscribir:
    def test_transcribe_audio_con_whisper_y_normaliza_segmentos(
        self, tmp_path, monkeypatch
    ):
        """transcribir debe cargar Whisper, llamar transcribe y normalizar."""
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake audio")
        resultado_whisper = {
            "text": " Hola mundo. Segundo segmento. ",
            "segments": [
                {
                    "start": 0,
                    "end": 1.5,
                    "text": " Hola mundo. ",
                    "words": [
                        {"word": " Hola", "start": 0.0, "end": 0.6, "probability": 0.91},
                        {"word": " mundo.", "start": 0.6, "end": 1.5, "probability": 0.88},
                    ],
                },
                {
                    "start": 1.5,
                    "end": 3,
                    "text": " Segundo segmento. ",
                    "words": [
                        {"word": " Segundo", "start": 1.5, "end": 2.1, "probability": 0.95},
                        {"word": " segmento.", "start": 2.1, "end": 3.0, "probability": 0.93},
                    ],
                },
            ],
        }
        fake_whisper = FakeWhisperModule(resultado_whisper)
        monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

        segmentos = transcribir(audio_path, modelo="large-v3", idioma="es")

        assert fake_whisper.modelo_cargado == "large-v3"
        assert fake_whisper.model.transcribe_calls == [
            {
                "audio_path": str(audio_path),
                "kwargs": {
                    "language": "es",
                    "word_timestamps": True,
                    "fp16": False,
                },
            }
        ]
        assert segmentos == [
            SegmentoRaw(
                texto="Hola mundo.",
                t_start=0.0,
                t_end=1.5,
                pausa_antes=0.0,
                words=[
                    WordTimestamp(word="Hola", start=0.0, end=0.6, probability=0.91),
                    WordTimestamp(word="mundo.", start=0.6, end=1.5, probability=0.88),
                ],
            ),
            SegmentoRaw(
                texto="Segundo segmento.",
                t_start=1.5,
                t_end=3.0,
                pausa_antes=0.0,
                words=[
                    WordTimestamp(word="Segundo", start=1.5, end=2.1, probability=0.95),
                    WordTimestamp(word="segmento.", start=2.1, end=3.0, probability=0.93),
                ],
            ),
        ]

    def test_ignora_segmentos_sin_texto(self, tmp_path, monkeypatch):
        """Segmentos vacíos o whitespace no deben pasar al siguiente componente."""
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake audio")
        resultado_whisper = {
            "segments": [
                {"start": 0, "end": 1, "text": "   "},
                {"start": 1, "end": 2, "text": " Texto válido. "},
            ]
        }
        monkeypatch.setitem(sys.modules, "whisper", FakeWhisperModule(resultado_whisper))

        segmentos = transcribir(audio_path)

        assert segmentos == [
            SegmentoRaw(
                texto="Texto válido.",
                t_start=1.0,
                t_end=2.0,
                pausa_antes=0.0,
                words=[],
            ),
        ]

    def test_audio_inexistente_lanza_file_not_found(self, tmp_path):
        """Si el archivo de audio no existe, falla antes de cargar Whisper."""
        audio_path = tmp_path / "no_existe.wav"

        with pytest.raises(FileNotFoundError, match="Audio no encontrado"):
            transcribir(audio_path)

    def test_pausa_antes_se_calcula_con_gap(self, tmp_path, monkeypatch):
        """Cuando hay un gap entre segmentos, pausa_antes debe reflejarlo."""
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake audio")
        resultado_whisper = {
            "segments": [
                {"start": 0, "end": 2.0, "text": "Primer segmento."},
                {"start": 5.0, "end": 7.0, "text": "Segundo segmento tras pausa."},
            ]
        }
        monkeypatch.setitem(sys.modules, "whisper", FakeWhisperModule(resultado_whisper))

        segmentos = transcribir(audio_path)

        assert segmentos[0].pausa_antes == 0.0
        assert segmentos[1].pausa_antes == 3.0


# ──────────────────────────────────────────────────────────
# Tests: guardar_transcripcion
# ──────────────────────────────────────────────────────────

class TestGuardarTranscripcion:
    def test_guarda_json_y_devuelve_ruta(self, transcript_mock, tmp_path):
        ruta = guardar_transcripcion(
            transcript_mock, "TestPlaylist", base_dir=tmp_path
        )

        assert ruta.exists()
        assert ruta.name == "vid001.json"
        assert "transcripciones-TestPlaylist" in str(ruta)

    def test_crea_directorio_si_no_existe(self, transcript_mock, tmp_path):
        ruta = guardar_transcripcion(
            transcript_mock, "TestPlaylist", base_dir=tmp_path
        )

        assert ruta.parent.exists()

    def test_json_contiene_video_id_y_segments(self, transcript_mock, tmp_path):
        ruta = guardar_transcripcion(
            transcript_mock, "TestPlaylist", base_dir=tmp_path
        )

        data = json.loads(ruta.read_text(encoding="utf-8"))
        assert data["video_id"] == "vid001"
        assert data["titulo"] == "Video Uno"
        assert data["fecha"] == "20260101"
        assert len(data["raw_segments"]) == 2
        assert data["raw_segments"][0]["texto"] == "Hola mundo."
        assert data["raw_segments"][0]["pausa_antes"] == 0.0
        assert data["raw_segments"][1]["pausa_antes"] == 3.5
        assert len(data["raw_segments"][0]["words"]) == 2
        assert data["raw_segments"][0]["words"][0]["word"] == "Hola"

    def test_sobrescribe_si_ya_existe(self, transcript_mock, tmp_path):
        guardar_transcripcion(transcript_mock, "TestPlaylist", base_dir=tmp_path)
        ruta = guardar_transcripcion(
            transcript_mock, "TestPlaylist", base_dir=tmp_path
        )

        assert ruta.exists()
        data = json.loads(ruta.read_text(encoding="utf-8"))
        assert data["video_id"] == "vid001"


# ──────────────────────────────────────────────────────────
# Tests: cargar_transcripcion
# ──────────────────────────────────────────────────────────

class TestCargarTranscripcion:
    def test_roundtrip_guardar_cargar_igualdad(self, transcript_mock, tmp_path):
        ruta = guardar_transcripcion(
            transcript_mock, "TestPlaylist", base_dir=tmp_path
        )
        cargado = cargar_transcripcion(ruta)

        assert cargado == transcript_mock

    def test_archivo_inexistente_lanza_error(self, tmp_path):
        ruta = tmp_path / "no_existe.json"
        with pytest.raises(FileNotFoundError):
            cargar_transcripcion(ruta)

    def test_json_corrupto_lanza_error(self, tmp_path):
        ruta = tmp_path / "roto.json"
        ruta.write_text("{json roto", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON inválido"):
            cargar_transcripcion(ruta)
